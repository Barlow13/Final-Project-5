/**
 * Pico-RoadDetector: Main Application
 *
 * This application uses a Raspberry Pi Pico to run TensorFlow Lite machine learning
 * models for road object detection. It captures images from an ArduCAM camera,
 * processes them through a neural network, and displays results on an OLED screen.
 *
 * MEMORY OPTIMIZED VERSION
 */
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/i2c.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "pico/time.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "picojpeg.h"
#include "jpeg_decoder.h"

// Project headers
#include "hardware_config.h"
#include "ov5642_regs.h"
#include "ssd1306.h"
#include "pca9546.h"
#include "class_names.h"
#include "model_data.h"

// TensorFlow Lite includes
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Debug mode
#define DEBUG 1

// Reduced model dimensions for lower memory usage
#define MODEL_WIDTH 64
#define MODEL_HEIGHT 64

// Global variables
ArduCAM myCAM(OV5642, PIN_CS);
uint8_t *shared_buffer = NULL;
#define SHARED_BUFFER_SIZE 32768 // 32KB shared buffer

const float class_thresholds[7] = {
    0.19, // bicycle - LOWER to improve extremely low precision
    0.30, // car - INCREASE slightly to reduce false positives
    0.25, // motorcycle - DECREASE to improve poor recall
    0.28, // bus - KEEP (good balance)
    0.27, // truck - INCREASE to reduce many false positives
    0.23, // traffic light - INCREASE to reduce false positives
    0.25  // stop sign - DECREASE to improve recall
};

// TensorFlow Lite globals
namespace
{
    // TF Lite error reporter
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    // Model and interpreter
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;

    // Input and output tensors
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    // Reduced tensor arena size
    constexpr int kTensorArenaSize = 128 * 1024; // 128KB tensor arena
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Function prototypes
void debug_print(const char *msg);
bool setup_hardware();
bool setup_camera();
bool setup_display();
bool setup_tensorflow();
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size);
bool process_image_for_inference(uint8_t *raw_buffer, uint32_t raw_size);
void display_results(const uint8_t *results, int num_classes, uint32_t inference_time);
bool run_inference();
void cleanup_resources();

/**
 * Helper function for debug output
 */
void debug_print(const char *msg)
{
#if DEBUG
    printf("%s\n", msg);
#endif
}

/**
 * Initialize hardware components
 */
bool setup_hardware()
{
    // Initialize stdio
    stdio_init_all();
    sleep_ms(1000); // Give USB time to initialize

    set_sys_clock_khz(125000, true);

    debug_print("Initializing hardware...");

    // Initialize I2C
    i2c_init(I2C_PORT, I2C_FREQ);
    gpio_set_function(PIN_SDA, GPIO_FUNC_I2C);
    gpio_set_function(PIN_SCL, GPIO_FUNC_I2C);
    gpio_pull_up(PIN_SDA);
    gpio_pull_up(PIN_SCL);

    // Initialize SPI
    spi_init(SPI_PORT, ARDUCAM_SPI_FREQ);
    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MISO, GPIO_FUNC_SPI);

    // Initialize CS pin
    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);

    printf("I2C and SPI initialized\n");

    // Initialize I2C multiplexer
    if (!pca9546_init(I2C_PORT, PCA9546_ADDR))
    {
        printf("Failed to initialize PCA9546 multiplexer\n");
        return false;
    }

    printf("PCA9546 multiplexer initialized\n");
    debug_print("Hardware initialized");
    return true;
}

/**
 * Initialize OLED display
 */
bool setup_display()
{
    debug_print("Setting up OLED display...");

    // Select OLED on multiplexer
    if (!pca9546_select(I2C_PORT, MUX_PORT_OLED))
    {
        printf("Failed to select OLED on multiplexer\n");
        return false;
    }

    // Initialize display
    ssd1306_init(I2C_PORT, SSD1306_ADDR);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "INITIALIZING...", 1);
    ssd1306_show();

    debug_print("OLED display ready");
    return true;
}

/**
 * Verify correct SPI communication with ArduCAM controller
 */
bool verify_arducam_spi()
{
    printf("Verifying ArduCAM SPI communication...\n");

    // Test write/read to a test register
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    sleep_ms(5);
    uint8_t read_val = myCAM.read_reg(ARDUCHIP_TEST1);
    printf("Write 0x55, read: 0x%02X\n", read_val);

    if (read_val != 0x55)
    {
        printf("SPI communication test failed! Expected 0x55, got 0x%02X\n", read_val);
        return false;
    }

    printf("ArduCAM SPI communication verified\n");
    return true;
}

/**
 * Verify camera sensor is responding
 */
bool verify_camera_sensor()
{
    // Select ArduCAM channel on multiplexer
    if (!pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM))
    {
        printf("Failed to select ArduCAM channel on multiplexer\n");
        return false;
    }

    // Software reset the sensor
    myCAM.wrSensorReg16_8(0x3008, 0x80);
    sleep_ms(100);

    // Clear reset flag
    myCAM.wrSensorReg16_8(0x3008, 0x00);
    sleep_ms(100);

    // Check camera ID
    uint8_t vid, pid;
    myCAM.wrSensorReg16_8(0xFF, 0x01);
    myCAM.rdSensorReg16_8(OV5642_CHIPID_HIGH, &vid);
    myCAM.rdSensorReg16_8(OV5642_CHIPID_LOW, &pid);

    printf("Camera ID check - VID: 0x%02X, PID: 0x%02X\n", vid, pid);

    if (vid != 0x56 || pid != 0x42)
    {
        printf("Camera ID mismatch! Expected VID=0x56, PID=0x42\n");
        return false;
    }

    printf("OV5642 camera sensor verified\n");
    return true;
}

/**
 * Test camera capture to verify functionality
 */
bool test_camera_capture()
{
    printf("Testing camera capture...\n");

    // Force JPEG mode for test capture
    myCAM.wrSensorReg16_8(0x4300, 0x18);
    sleep_ms(10);

    // Reset FIFO before capture
    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();
    sleep_ms(50);

    // Start capture
    myCAM.start_capture();
    printf("Test capture started\n");

    // Wait for capture with timeout
    bool capture_done = false;
    uint32_t start_time = to_ms_since_boot(get_absolute_time());

    while (!capture_done)
    {
        if (myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK))
        {
            capture_done = true;
            printf("Test capture completed in %d ms\n",
                   (int)(to_ms_since_boot(get_absolute_time()) - start_time));
            break;
        }

        if (to_ms_since_boot(get_absolute_time()) - start_time > 3000)
        {
            printf("Test capture timed out after 3 seconds\n");
            return false;
        }

        sleep_ms(10);
    }

    // Read FIFO length
    uint32_t length = myCAM.read_fifo_length();
    printf("Test image size: %lu bytes\n", length);

    // Check if image size is reasonable
    if (length < 1000 || length > 200000)
    {
        printf("Test image size is unreasonable: %lu bytes\n", length);
        return false;
    }

    // Allocate buffer for the JPEG header to check validity
    const size_t HEADER_SIZE = 32;
    uint8_t header[HEADER_SIZE];

    // Read image header data from FIFO
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();

    // Read JPEG header to confirm it's valid
    for (int i = 0; i < HEADER_SIZE && i < length; i++)
    {
        spi_read_blocking(SPI_PORT, 0, &header[i], 1);
    }

    myCAM.CS_HIGH();

    // Print first bytes for debugging
    printf("JPEG header: ");
    for (int i = 0; i < 16; i++)
    {
        printf("%02X ", header[i]);
    }
    printf("\n");

    // Check for JPEG signature (0xFF 0xD8)
    bool valid_jpeg = (header[0] == 0xFF && header[1] == 0xD8);

    if (!valid_jpeg)
    {
        printf("Invalid JPEG header\n");
        return false;
    }

    printf("Valid JPEG data captured - Camera test PASSED\n");

    // Clear FIFO flag to complete test
    myCAM.clear_fifo_flag();
    return true;
}

/**
 * Initialize ArduCAM camera
 */
bool setup_camera()
{
    debug_print("Setting up ArduCAM...");

    // Initialize ArduCAM
    myCAM.Arducam_init();
    sleep_ms(100);

    // Select ArduCAM channel on multiplexer
    if (!pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM))
    {
        printf("Failed to select ArduCAM channel on multiplexer\n");
        return false;
    }

    // Reset hardware
    myCAM.write_reg(0x07, 0x80);
    sleep_ms(100);
    myCAM.write_reg(0x07, 0x00);
    sleep_ms(100);

    // Verify SPI communication
    if (!verify_arducam_spi())
    {
        printf("SPI communication verification failed\n");
        return false;
    }

    // Verify camera sensor
    if (!verify_camera_sensor())
    {
        printf("Camera sensor verification failed\n");
        return false;
    }

    // Initialize camera mode and settings
    myCAM.set_format(JPEG);
    myCAM.InitCAM();
    sleep_ms(50);

    // Configure timing
    myCAM.write_reg(ARDUCHIP_TIM, VSYNC_LEVEL_MASK);
    sleep_ms(50);

    // Set lower resolution for memory savings
    myCAM.OV5642_set_JPEG_size(OV5642_64x64);
    sleep_ms(100);

    // Reset FIFO
    myCAM.clear_fifo_flag();
    myCAM.write_reg(ARDUCHIP_FRAMES, 0x00);
    sleep_ms(100);

    // Set critical JPEG registers
    myCAM.wrSensorReg16_8(0x4300, 0x18); // Format control - YUV422 + JPEG
    myCAM.wrSensorReg16_8(0x3818, 0xA8); // Timing control
    myCAM.wrSensorReg16_8(0x3621, 0x10); // Array control
    myCAM.wrSensorReg16_8(0x3801, 0xB0); // Timing HS
    myCAM.wrSensorReg16_8(0x4407, 0x04); // Compression quantization
    sleep_ms(100);

    // Perform test capture to verify all is working
    if (!test_camera_capture())
    {
        printf("Camera capture test failed\n");
        return false;
    }

    debug_print("ArduCAM camera initialized successfully");
    return true;
}

/**
 * Initialize TensorFlow Lite
 */
bool setup_tensorflow()
{
    debug_print("Setting up TensorFlow Lite...");

    // Update display with status
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "LOADING MODEL", 1);
    ssd1306_show();

    // Print memory info
    printf("Tensor arena size: %d KB\n", kTensorArenaSize / 1024);

    // Map the model into a usable data structure
    model = tflite::GetModel(model_data);

    if (!model)
    {
        printf("ERROR: Failed to get model\n");
        return false;
    }

    printf("Model version: %lu, Schema version: %d\n", model->version(), TFLITE_SCHEMA_VERSION);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report("Model schema version mismatch!");
        return false;
    }

    // Create a resolver with sufficient capacity for all operations
    static tflite::MicroMutableOpResolver<16> resolver;

    // Register operations needed by the model
    resolver.AddConv2D();          // Convolutions
    resolver.AddDepthwiseConv2D(); // MobileNetV2 depthwise convs
    resolver.AddFullyConnected();  // Dense layers
    resolver.AddReshape();         // Reshape operations
    resolver.AddSoftmax();         // Final classification
    resolver.AddAdd();             // Skip connections and residuals
    resolver.AddMul();             // Various operations
    resolver.AddAveragePool2D();   // Pooling layers
    resolver.AddMaxPool2D();       // Max pooling operations
    resolver.AddMean();            // Global average pooling
    resolver.AddQuantize();        // Quantization operations
    resolver.AddDequantize();      // Dequantization operations
    resolver.AddPad();             // Padding operations
    resolver.AddConcatenation();   // Feature concatenation
    resolver.AddRelu6();           // ReLU6 activations in MobileNetV2
    resolver.AddLogistic();        // Sigmoid/logistic for binary classification

    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();

    if (allocate_status != kTfLiteOk)
    {
        error_reporter->Report("AllocateTensors() failed with status %d", allocate_status);
        return false;
    }

    // Get pointers to input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print tensor information for debugging
    printf("Input tensor type: %d (kTfLiteInt8=%d, kTfLiteUInt8=%d, kTfLiteFloat32=%d)\n",
           input->type, kTfLiteInt8, kTfLiteUInt8, kTfLiteFloat32);
    printf("Output tensor type: %d\n", output->type);
    printf("Input tensor dims: %d x %d x %d x %d\n",
           input->dims->data[0], input->dims->data[1],
           input->dims->data[2], input->dims->data[3]);
    printf("Output tensor dims: %d x %d\n",
           output->dims->data[0], output->dims->data[1]);
    printf("Arena size: %d bytes used, %d bytes available\n",
           interpreter->arena_used_bytes(), kTensorArenaSize);

    // Display success
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "MODEL READY", 1);
    ssd1306_show();
    sleep_ms(500);

    debug_print("TensorFlow Lite ready");
    return true;
}

/**
 * Capture image from camera and store in buffer
 *
 * @param buffer Pointer to buffer for image storage
 * @param buffer_size Size of the buffer
 * @param captured_size Pointer to variable that will store the captured size
 * @return true if successful, false otherwise
 */
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size)
{
    // Select ArduCAM on multiplexer
    if (!pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM))
    {
        printf("Failed to select ArduCAM on multiplexer for capture\n");
        return false;
    }

    // Force critical JPEG registers before capture
    myCAM.wrSensorReg16_8(0x4300, 0x18); // Format control - YUV422 + JPEG
    myCAM.wrSensorReg16_8(0x501F, 0x00); // ISP output format
    myCAM.wrSensorReg16_8(0x3818, 0xA8); // Timing control
    myCAM.wrSensorReg16_8(0x3621, 0x10); // Array control
    myCAM.wrSensorReg16_8(0x3801, 0xB0); // Timing HS
    myCAM.wrSensorReg16_8(0x4407, 0x04); // Quantization scale
    sleep_ms(50);

    // Reset FIFO
    myCAM.flush_fifo();
    sleep_ms(5);
    myCAM.clear_fifo_flag();
    sleep_ms(5);

    // Start capture
    printf("Starting image capture...\n");
    myCAM.start_capture();

    // Wait for capture with timeout
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    bool capture_timeout = false;

    while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK))
    {
        if (to_ms_since_boot(get_absolute_time()) - start_time > 3000)
        {
            capture_timeout = true;
            break;
        }
        sleep_ms(10);
    }

    if (capture_timeout)
    {
        printf("Error: Camera capture timeout\n");
        return false;
    }

    printf("Capture completed successfully\n");

    // Read captured data size
    uint32_t length = myCAM.read_fifo_length();
    printf("FIFO length: %lu bytes\n", length);

    if (length > buffer_size || length < 20)
    {
        printf("Error: Image size invalid or too large for buffer: %lu bytes\n", length);
        return false;
    }

    // Store captured size
    *captured_size = length;

    // Read image data from FIFO into buffer
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();

    // Read data in chunks for better stability
    const size_t CHUNK_SIZE = 1024;
    for (uint32_t i = 0; i < length; i += CHUNK_SIZE)
    {
        size_t bytes_to_read = (i + CHUNK_SIZE > length) ? (length - i) : CHUNK_SIZE;
        for (size_t j = 0; j < bytes_to_read; j++)
        {
            spi_read_blocking(SPI_PORT, 0, &buffer[i + j], 1);
        }
    }

    myCAM.CS_HIGH();

    // Print first few bytes for debugging
    printf("First 16 bytes: ");
    for (int i = 0; i < 16 && i < length; i++)
    {
        printf("%02X ", buffer[i]);
    }
    printf("\n");

    // Verify JPEG header
    if (buffer[0] != 0xFF || buffer[1] != 0xD8)
    {
        printf("Error: Invalid JPEG header\n");
        return false;
    }

    // Reset FIFO flag
    myCAM.clear_fifo_flag();

    return true;
}

/**
 * Process image data for inference
 *
 * @param raw_buffer Raw JPEG image data
 * @param raw_size Size of the raw data
 * @return true if successful, false otherwise
 */
bool process_image_for_inference(uint8_t *raw_buffer, uint32_t raw_size)
{
    // Allocate shared buffer if not already allocated
    if (!shared_buffer)
    {
        shared_buffer = (uint8_t *)malloc(SHARED_BUFFER_SIZE);
        if (!shared_buffer)
        {
            printf("Error: Failed to allocate shared buffer\n");
            return false;
        }
    }

    // Decode to grayscale at reduced resolution directly into shared buffer
    if (!jpeg_decode_to_model_input(raw_buffer, raw_size,
                                    shared_buffer,
                                    MODEL_WIDTH, MODEL_HEIGHT))
    {
        printf("Error: JPEG decoding failed\n");
        return false;
    }

    // Handle different tensor types
    printf("Input tensor type: %d (UInt8=%d, Float32=%d, Int8=%d)\n",
           input->type, kTfLiteUInt8, kTfLiteFloat32, kTfLiteInt8);

    if (input->type == kTfLiteInt8)
    {
        // For int8 quantized model input - using exact parameters from training
        int8_t *input_data = input->data.int8;
        float scale = 0.003922f; // 1/255 from training
        int zero_point = -128;   // From training

        printf("Using scale=%f and zero_point=%d for INT8 tensors\n", scale, zero_point);

        for (int y = 0; y < MODEL_HEIGHT; y++)
        {
            for (int x = 0; x < MODEL_WIDTH; x++)
            {
                // Get pixel value from our decoded grayscale image
                uint8_t pixel = shared_buffer[y * MODEL_WIDTH + x];

                // Convert to INT8 range (-128 to 127) using training parameters
                int8_t quantized = (int8_t)(pixel - 128);

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3)
                {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = quantized; // R
                    input_data[dst_idx + 1] = quantized; // G
                    input_data[dst_idx + 2] = quantized; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1)
                {
                    input_data[y * MODEL_WIDTH + x] = quantized;
                }
            }
        }
    }
    else if (input->type == kTfLiteUInt8)
    {
        // For quantized model input (uint8)
        uint8_t *input_data = input->data.uint8;

        for (int y = 0; y < MODEL_HEIGHT; y++)
        {
            for (int x = 0; x < MODEL_WIDTH; x++)
            {
                // Get pixel value
                uint8_t pixel = shared_buffer[y * MODEL_WIDTH + x];

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3)
                {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel; // R
                    input_data[dst_idx + 1] = pixel; // G
                    input_data[dst_idx + 2] = pixel; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1)
                {
                    input_data[y * MODEL_WIDTH + x] = pixel;
                }
            }
        }
    }
    else if (input->type == kTfLiteFloat32)
    {
        // For floating point model input (float32)
        float *input_data = input->data.f;

        for (int y = 0; y < MODEL_HEIGHT; y++)
        {
            for (int x = 0; x < MODEL_WIDTH; x++)
            {
                // Get pixel value and normalize to 0-1 range
                float pixel = shared_buffer[y * MODEL_WIDTH + x] / 255.0f;

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3)
                {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel; // R
                    input_data[dst_idx + 1] = pixel; // G
                    input_data[dst_idx + 2] = pixel; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1)
                {
                    input_data[y * MODEL_WIDTH + x] = pixel;
                }
            }
        }
    }
    else
    {
        printf("Error: Unsupported input tensor type: %d\n", input->type);
        return false;
    }

    return true;
}

/**
 * Enhanced OLED display: 
 *   Line 1: All detected classes (comma separated)
 *   Line 2: Highest confidence% and inference time
 */
void display_results(const uint8_t *results, int num_classes, uint32_t inference_time) {
    // Keep track of highest confidence class for second line
    int best_idx = 0;
    int best_score = results[0];
    
    // Find the highest confidence class
    for (int i = 1; i < num_classes; i++) {
        if (results[i] > best_score) {
            best_score = results[i];
            best_idx = i;
        }
    }
    
    // Convert highest confidence to percentage
    int confidence_pct = (best_score * 100) / 255;
    
    // Build a string with all detected classes
    char detected_classes[32] = ""; // Buffer for detected classes
    int detected_count = 0;
    
    for (int i = 0; i < num_classes; i++) {
        // Calculate percentage confidence
        int class_conf = (results[i] * 100) / 255;
        
        // Check if confidence exceeds class threshold
        float threshold_pct = class_thresholds[i] * 100.0f;
        if (class_conf >= (int)threshold_pct) {
            // If not the first detection, add comma separator
            if (detected_count > 0) {
                strcat(detected_classes, ",");
            }
            
            // Add class name
            strcat(detected_classes, class_names[i]);
            detected_count++;
        }
    }
    
    // If nothing detected, show "NONE"
    if (detected_count == 0) {
        strcpy(detected_classes, "NONE");
    }
    
    // Prepare second line (e.g. "87% 45ms")
    char info[17];
    snprintf(info, sizeof(info), "%d%% %dms", confidence_pct, (int)inference_time);
    
    // Render to screen
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, detected_classes, 1);  // top line - all detections
    ssd1306_draw_string(0, 16, info, 1);             // bottom line - highest conf + time
    ssd1306_show();
    
    // Print detections to console for debugging
    printf("Detections: %s (best: %s at %d%%)\n", 
           detected_classes, 
           detected_count > 0 ? class_names[best_idx] : "none", 
           confidence_pct);
}


/**
 * Run inference on a captured image
 */
bool run_inference()
{
    // Update display
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "CAPTURING...", 1);
    ssd1306_show();

    // Allocate memory for camera capture
    const size_t RAW_BUFFER_SIZE = 30000; // Reduced from 60000
    uint8_t *raw_buffer = (uint8_t *)malloc(RAW_BUFFER_SIZE);

    if (!raw_buffer)
    {
        printf("Error: Failed to allocate memory for raw image data\n");
        return false;
    }

    // Capture image
    uint32_t captured_size = 0;
    bool capture_success = capture_image_to_buffer(raw_buffer, RAW_BUFFER_SIZE, &captured_size);

    if (!capture_success)
    {
        printf("Error: Image capture failed\n");
        free(raw_buffer);

        // Show error on display
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "CAPTURE ERROR", 1);
        ssd1306_show();
        sleep_ms(1000);

        return false;
    }

    // Update display
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "PROCESSING...", 1);
    ssd1306_show();

    // Process image for inference
    if (!process_image_for_inference(raw_buffer, captured_size))
    {
        printf("Error: Image processing failed\n");
        free(raw_buffer);

        // Show error on display
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "PROCESS ERROR", 1);
        ssd1306_show();
        sleep_ms(1000);

        return false;
    }

    // Free raw buffer as it's no longer needed
    free(raw_buffer);

    // Update display
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "INFERENCING...", 1);
    ssd1306_show();

    // Measure inference time
    uint32_t start_time = to_ms_since_boot(get_absolute_time());

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();

    uint32_t inference_time = to_ms_since_boot(get_absolute_time()) - start_time;
    printf("Inference took %d ms\n", (int)inference_time);

    if (invoke_status != kTfLiteOk)
    {
        printf("Inference failed with status: %d\n", invoke_status);

        // Show error on display
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "INFERENCE ERROR", 1);
        ssd1306_show();
        sleep_ms(1000);

        return false;
    }

    // Process results based on output tensor type
    if (output->type == kTfLiteUInt8)
    {
        uint8_t *results = output->data.uint8;
        int num_classes = output->dims->data[1];
        display_results(results, num_classes, inference_time);
    }
    else if (output->type == kTfLiteInt8)
    {
        // For int8 output, we need to dequantize manually
        int8_t *int8_results = output->data.int8;
        int num_classes = output->dims->data[1];
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;

        // Use shared buffer for conversion
        uint8_t *uint8_results = shared_buffer;
        if (!uint8_results)
        {
            printf("Failed to allocate memory for results conversion\n");
            return false;
        }

        // Convert from int8 to uint8 with proper dequantization
        for (int i = 0; i < num_classes; i++)
        {
            float dequantized = scale * (int8_results[i] - zero_point);
            // Clip to 0-1 range and convert to 0-255
            dequantized = dequantized < 0.0f ? 0.0f : (dequantized > 1.0f ? 1.0f : dequantized);
            uint8_results[i] = (uint8_t)(dequantized * 255.0f);
        }

        display_results(uint8_results, num_classes, inference_time);
    }
    else
    {
        printf("Unsupported output tensor type: %d\n", output->type);
        return false;
    }

    return true;
}

/**
 * Clean up resources
 */
void cleanup_resources()
{
    if (shared_buffer)
    {
        free(shared_buffer);
        shared_buffer = NULL;
    }
}

/**
 * Main application entry point
 */
int main()
{

    // Initialize all components
    if (!setup_hardware())
    {
        printf("Hardware setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    if (!setup_display())
    {
        printf("Display setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    if (!setup_camera())
    {
        printf("Camera setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    if (!setup_tensorflow())
    {
        printf("TensorFlow setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    // Show ready message
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "SYSTEM READY", 1);
    ssd1306_show();
    sleep_ms(1000);

    // Main loop
    while (true)
    {
        run_inference();
        sleep_ms(2000);
    }

    // Clean up resources (never reached in this loop)
    cleanup_resources();

    return 0;
}