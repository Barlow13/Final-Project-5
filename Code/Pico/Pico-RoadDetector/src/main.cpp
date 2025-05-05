/**
 * Pico-RoadDetector: Optimized Dual-Core Implementation
 *
 * This application uses both RP2040 cores to efficiently run TensorFlow Lite
 * machine learning models for road object detection:
 * - Core 0: I/O operations (camera, display, user interface)
 * - Core 1: TensorFlow Lite inference and image processing
 */

#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "hardware/i2c.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "hardware/sync.h"
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

// Model dimensions
#define MODEL_WIDTH 64
#define MODEL_HEIGHT 64

// Spinlock for shared memory access
spin_lock_t *memory_lock;
#define MEMORY_LOCK_ID 0

// Queue for inter-core commands
queue_t core0_to_core1_queue;
queue_t core1_to_core0_queue;

// Commands for inter-core communication
enum CoreCommand {
    CMD_PROCESS_IMAGE = 1,
    CMD_INFERENCE_COMPLETE = 2,
    CMD_ERROR = 3
};

// Shared data structure for inference results
struct InferenceResult {
    float scores[7];
    uint8_t predictions[7];
    uint32_t inference_time_ms;
    bool valid;
};

// Pre-allocated buffers to reduce memory fragmentation
// Buffer for camera capture - shared between cores
static uint8_t g_capture_buffer[32768] __attribute__((aligned(8)));
static uint32_t g_capture_size = 0;

// Buffer for image processing - used by core 1
static uint8_t g_process_buffer[MODEL_WIDTH * MODEL_HEIGHT * 3] __attribute__((aligned(8)));

// Results shared between cores
static volatile InferenceResult g_inference_result __attribute__((aligned(8)));

// Optimized thresholds from first version based on evaluation
const float g_improved_thresholds[7] = {
    0.2105, // bicycle
    0.3211, // car
    0.2474, // motorcycle
    0.2842, // bus
    0.2474, // truck
    0.2474, // traffic light
    0.2474  // stop sign
};

// Global variables for hardware
ArduCAM myCAM(OV5642, PIN_CS);

// Forward declarations
bool setup_hardware();
bool setup_camera();
bool setup_display();
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size);
void debug_print(const char *msg);
void display_message(const char *line1, const char *line2, const char *line3);

// TensorFlow Lite globals for Core 1
namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    constexpr int kTensorArenaSize = 128 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Helper function for debug output
void debug_print(const char *msg) {
#if DEBUG
    printf("%s\n", msg);
#endif
}

// Display a message on the OLED screen with up to three rows of text
void display_message(const char *line1, const char *line2 = nullptr, const char *line3 = nullptr) {
    // Select OLED on multiplexer
    int result = pca9546_select(I2C_PORT, MUX_PORT_OLED);
    if (result != PCA9546_SUCCESS) {
        printf("Failed to select OLED on multiplexer: %s\n", 
               pca9546_error_string(result));
        return;
    }
    
    ssd1306_clear();
    
    // Display first line
    if (line1) {
        ssd1306_draw_string(0, 0, line1, 1);
    }
    
    // Display second line
    if (line2) {
        ssd1306_draw_string(0, 16, line2, 1);
    }
    
    // Display third line
    if (line3) {
        ssd1306_draw_string(0, 32, line3, 1);
    }
    
    ssd1306_show();
}

// Initialize hardware components
bool setup_hardware() {
    // Initialize stdio
    stdio_init_all();
    sleep_ms(1000); // Give USB time to initialize

    // Set CPU clock to maximum
    set_sys_clock_khz(133000, true);

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
    int mux_result = pca9546_init(I2C_PORT, PCA9546_ADDR);
    if (mux_result != PCA9546_SUCCESS) {
        printf("Failed to initialize PCA9546 multiplexer: %s\n", 
               pca9546_error_string(mux_result));
        return false;
    }

    printf("PCA9546 multiplexer initialized\n");
    
    // Initialize spinlock for memory protection
    memory_lock = spin_lock_init(MEMORY_LOCK_ID);
    
    // Initialize queues for inter-core communication
    queue_init(&core0_to_core1_queue, sizeof(uint32_t), 4);
    queue_init(&core1_to_core0_queue, sizeof(uint32_t), 4);
    
    debug_print("Hardware initialized");
    return true;
}

// Initialize OLED display
bool setup_display() {
    debug_print("Setting up OLED display...");

    // Select OLED on multiplexer
    int result = pca9546_select(I2C_PORT, MUX_PORT_OLED);
    if (result != PCA9546_SUCCESS) {
        printf("Failed to select OLED on multiplexer: %s\n", 
               pca9546_error_string(result));
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

// Verify correct SPI communication with ArduCAM controller
bool verify_arducam_spi() {
    printf("Verifying ArduCAM SPI communication...\n");

    // Test write/read to a test register
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    sleep_ms(5);
    uint8_t read_val = myCAM.read_reg(ARDUCHIP_TEST1);
    printf("Write 0x55, read: 0x%02X\n", read_val);

    if (read_val != 0x55) {
        printf("SPI communication test failed! Expected 0x55, got 0x%02X\n", read_val);
        return false;
    }

    printf("ArduCAM SPI communication verified\n");
    return true;
}

// Verify camera sensor is responding
bool verify_camera_sensor() {
    // Select ArduCAM channel on multiplexer
    int result = pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM);
    if (result != PCA9546_SUCCESS) {
        printf("Failed to select ArduCAM channel on multiplexer: %s\n", 
               pca9546_error_string(result));
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

    if (vid != 0x56 || pid != 0x42) {
        printf("Camera ID mismatch! Expected VID=0x56, PID=0x42\n");
        return false;
    }

    printf("OV5642 camera sensor verified\n");
    return true;
}

// Test camera capture to verify functionality
bool test_camera_capture() {
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

    while (!capture_done) {
        if (myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
            capture_done = true;
            printf("Test capture completed in %d ms\n",
                   (int)(to_ms_since_boot(get_absolute_time()) - start_time));
            break;
        }

        if (to_ms_since_boot(get_absolute_time()) - start_time > 3000) {
            printf("Test capture timed out after 3 seconds\n");
            return false;
        }

        sleep_ms(10);
    }

    // Read FIFO length
    uint32_t length = myCAM.read_fifo_length();
    printf("Test image size: %lu bytes\n", length);

    // Check if image size is reasonable
    if (length < 1000 || length > 200000) {
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
    for (int i = 0; i < HEADER_SIZE && i < length; i++) {
        spi_read_blocking(SPI_PORT, 0, &header[i], 1);
    }

    myCAM.CS_HIGH();

    // Print first bytes for debugging
    printf("JPEG header: ");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", header[i]);
    }
    printf("\n");

    // Check for JPEG signature (0xFF 0xD8)
    bool valid_jpeg = (header[0] == 0xFF && header[1] == 0xD8);

    if (!valid_jpeg) {
        printf("Invalid JPEG header\n");
        return false;
    }

    printf("Valid JPEG data captured - Camera test PASSED\n");

    // Clear FIFO flag to complete test
    myCAM.clear_fifo_flag();
    return true;
}

// Initialize ArduCAM camera
bool setup_camera() {
    debug_print("Setting up ArduCAM...");

    // Initialize ArduCAM
    myCAM.Arducam_init();
    sleep_ms(100);

    // Select ArduCAM channel on multiplexer
    int result = pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM);
    if (result != PCA9546_SUCCESS) {
        printf("Failed to select ArduCAM channel on multiplexer: %s\n", 
               pca9546_error_string(result));
        return false;
    }

    // Reset hardware
    myCAM.write_reg(0x07, 0x80);
    sleep_ms(100);
    myCAM.write_reg(0x07, 0x00);
    sleep_ms(100);

    // Verify SPI communication
    if (!verify_arducam_spi()) {
        printf("SPI communication verification failed\n");
        return false;
    }

    // Verify camera sensor
    if (!verify_camera_sensor()) {
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

    myCAM.OV5642_set_Mirror_Flip(FLIP);
    sleep_ms(200);

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
    if (!test_camera_capture()) {
        printf("Camera capture test failed\n");
        return false;
    }

    debug_print("ArduCAM camera initialized successfully");
    return true;
}

// Capture image from camera and store in buffer
// Optimized to reduce processing time
bool capture_image_to_buffer(uint8_t *buffer, size_t buffer_size, uint32_t *captured_size) {
    // Select ArduCAM on multiplexer
    int result = pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM);
    if (result != PCA9546_SUCCESS) {
        printf("Failed to select ArduCAM on multiplexer for capture: %s\n", 
               pca9546_error_string(result));
        return false;
    }

    // Force critical JPEG registers before capture
    myCAM.wrSensorReg16_8(0x4300, 0x18); // Format control - YUV422 + JPEG
    myCAM.wrSensorReg16_8(0x501F, 0x00); // ISP output format
    
    // Reset FIFO
    myCAM.flush_fifo();
    sleep_ms(5);
    myCAM.clear_fifo_flag();
    sleep_ms(5);

    // Start capture
    printf("Starting image capture...\n");
    absolute_time_t start_time = get_absolute_time();
    myCAM.start_capture();

    // Wait for capture with timeout
    bool capture_timeout = false;
    absolute_time_t timeout = make_timeout_time_ms(1000);  // 1-second timeout

    while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        if (absolute_time_diff_us(get_absolute_time(), timeout) <= 0) {
            capture_timeout = true;
            break;
        }
        sleep_us(100);  // Short sleep to be responsive but not waste CPU
    }

    if (capture_timeout) {
        printf("Error: Camera capture timeout\n");
        return false;
    }

    int elapsed_ms = absolute_time_diff_us(start_time, get_absolute_time()) / 1000;
    printf("Capture completed successfully in %d ms\n", elapsed_ms);

    // Read captured data size
    uint32_t length = myCAM.read_fifo_length();
    printf("FIFO length: %lu bytes\n", length);

    if (length > buffer_size || length < 20) {
        printf("Error: Image size invalid or too large for buffer: %lu bytes\n", length);
        return false;
    }

    // Store captured size
    *captured_size = length;

    // Read image data from FIFO into buffer
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();

    // Read data using DMA-optimized chunks for better performance
    const size_t CHUNK_SIZE = 1024;
    for (uint32_t i = 0; i < length; i += CHUNK_SIZE) {
        size_t bytes_to_read = (i + CHUNK_SIZE > length) ? (length - i) : CHUNK_SIZE;
        spi_read_blocking(SPI_PORT, 0, &buffer[i], bytes_to_read);
    }

    myCAM.CS_HIGH();

    // Print first few bytes for debugging
    printf("First 16 bytes: ");
    for (int i = 0; i < 16 && i < length; i++) {
        printf("%02X ", buffer[i]);
    }
    printf("\n");

    // Verify JPEG header
    if (buffer[0] != 0xFF || buffer[1] != 0xD8) {
        printf("Error: Invalid JPEG header\n");
        return false;
    }

    // Reset FIFO flag
    myCAM.clear_fifo_flag();

    return true;
}

// Process image for inference - runs on Core 1
bool process_image_for_inference(const uint8_t *raw_buffer, uint32_t raw_size, uint8_t *output_buffer) {
    // Decode JPEG to RGB at reduced resolution directly
    if (!jpeg_decode_to_model_input(raw_buffer, raw_size,
                                   output_buffer,
                                   MODEL_WIDTH, MODEL_HEIGHT)) {
        printf("Core 1: Error - JPEG decoding failed\n");
        return false;
    }
    
    return true;
}

// Fill TFLite input tensor with preprocessed image data
bool fill_input_tensor(const uint8_t *image_data) {
    if (!input) {
        printf("Error: Input tensor not initialized\n");
        return false;
    }

    // Handle different tensor types
    if (input->type == kTfLiteInt8) {
        // For int8 quantized model input
        int8_t *input_data = input->data.int8;
        float scale = 0.003922f; // 1/255 from training
        int zero_point = -128;   // From training

        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Get pixel value from our decoded grayscale image
                uint8_t pixel = image_data[y * MODEL_WIDTH + x];

                // Convert to INT8 range (-128 to 127) using training parameters
                int8_t quantized = (int8_t)(pixel - 128);

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = quantized; // R
                    input_data[dst_idx + 1] = quantized; // G
                    input_data[dst_idx + 2] = quantized; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1) {
                    input_data[y * MODEL_WIDTH + x] = quantized;
                }
            }
        }
    }
    else if (input->type == kTfLiteUInt8) {
        // For quantized model input (uint8)
        uint8_t *input_data = input->data.uint8;

        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Get pixel value
                uint8_t pixel = image_data[y * MODEL_WIDTH + x];

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel; // R
                    input_data[dst_idx + 1] = pixel; // G
                    input_data[dst_idx + 2] = pixel; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1) {
                    input_data[y * MODEL_WIDTH + x] = pixel;
                }
            }
        }
    }
    else if (input->type == kTfLiteFloat32) {
        // For floating point model input (float32)
        float *input_data = input->data.f;

        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Get pixel value and normalize to 0-1 range
                float pixel = image_data[y * MODEL_WIDTH + x] / 255.0f;

                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel; // R
                    input_data[dst_idx + 1] = pixel; // G
                    input_data[dst_idx + 2] = pixel; // B
                }
                // For single channel model inputs
                else if (input->dims->data[3] == 1) {
                    input_data[y * MODEL_WIDTH + x] = pixel;
                }
            }
        }
    }
    else {
        printf("Error: Unsupported input tensor type: %d\n", input->type);
        return false;
    }

    return true;
}

// Format and display detection results using three lines
// Enhanced to improve serial monitor and OLED display
void display_results(const InferenceResult *result) {
    if (!result || !result->valid) {
        display_message("DETECTION ERROR", "Invalid result", "");
        return;
    }
    
    // Keep track of highest confidence class for display
    int best_idx = 0;
    float best_score = result->scores[0];
    
    // Find the highest confidence class
    for (int i = 1; i < 7; i++) {
        if (result->scores[i] > best_score) {
            best_score = result->scores[i];
            best_idx = i;
        }
    }
    
    // Convert highest confidence to percentage
    int confidence_pct = (int)(best_score * 100);
    
    // Print individual class predictions to serial monitor
    printf("--------DETECTION RESULTS--------\n");
    for (int i = 0; i < 7; i++) {
        printf("Class: %-13s | Score: %5.2f%% | %s\n", 
               class_names[i], 
               result->scores[i] * 100.0f,
               result->predictions[i] ? "DETECTED" : "Not detected");
    }
    printf("Best detection: %s (%d%%)\n", class_names[best_idx], confidence_pct);
    printf("Inference time: %ld ms\n", result->inference_time_ms);
    printf("-------------------------------\n");
    
    // Build a string with detected classes (for line 1)
    char detected_classes[32] = ""; // Buffer for detected classes
    int detected_count = 0;
    
    for (int i = 0; i < 7; i++) {
        if (result->predictions[i]) {
            // If not the first detection, add comma separator
            if (detected_count > 0 && strlen(detected_classes) < 28) {
                strcat(detected_classes, ",");
            }
            
            // Add class name if there's enough space
            if (strlen(detected_classes) + strlen(class_names[i]) < 29) {
                strcat(detected_classes, class_names[i]);
                detected_count++;
            }
        }
    }
    
    // If nothing detected, show "NONE"
    if (detected_count == 0) {
        strcpy(detected_classes, "NONE DETECTED");
    }
    
    // Prepare second line - best detection with confidence
    char best_detection[17];
    if (detected_count > 0) {
        snprintf(best_detection, sizeof(best_detection), "%s: %d%%", 
                class_names[best_idx], confidence_pct);
    } else {
        snprintf(best_detection, sizeof(best_detection), "CONFIDENCE: %d%%", 
                confidence_pct);
    }
    
    // Prepare third line with timing info
    char timing_info[17];
    snprintf(timing_info, sizeof(timing_info), "INF: %ldms", result->inference_time_ms);
    
    // Select OLED on multiplexer before displaying
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
       
    // Display results on OLED
    ssd1306_clear();
    
    // Display first line (detected classes)
    ssd1306_draw_string(0, 0, detected_classes, 1);
    
    // Display second line (best detection with confidence)
    ssd1306_draw_string(0, 12, best_detection, 1);
    
    // Display third line (timing info)
    ssd1306_draw_string(0, 24, timing_info, 1);
    
    // Update display
    ssd1306_show();
}

// Core 1 entry function - handles all ML inference
void core1_entry() {
    printf("Core 1: Starting TensorFlow Lite initialization...\n");
    
    // Initialize TensorFlow Lite
    model = tflite::GetModel(model_data);
    if (!model) {
        printf("Core 1: ERROR - Failed to get TFLite model\n");
        uint32_t error_cmd = CMD_ERROR;
        queue_try_add(&core1_to_core0_queue, &error_cmd);
        return;
    }
    
    // Create resolver with all required operations
    static tflite::MicroMutableOpResolver<16> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddRelu6();
    resolver.AddLogistic();
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("Core 1: ERROR - Failed to allocate tensors: %d\n", allocate_status);
        uint32_t error_cmd = CMD_ERROR;
        queue_try_add(&core1_to_core0_queue, &error_cmd);
        return;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    printf("Core 1: TensorFlow Lite initialized successfully\n");
    printf("Core 1: Input tensor type: %d, dims: %d x %d x %d x %d\n",
           input->type, input->dims->data[0], input->dims->data[1],
           input->dims->data[2], input->dims->data[3]);
    printf("Core 1: Output tensor type: %d, dims: %d x %d\n",
           output->type, output->dims->data[0], output->dims->data[1]);
    
    // Signal that initialization is complete
    uint32_t init_complete = CMD_INFERENCE_COMPLETE;
    queue_try_add(&core1_to_core0_queue, &init_complete);
    
    // Main processing loop
    uint32_t command;
    
    while (true) {
        // Wait for a command from Core 0
        if (queue_try_remove(&core0_to_core1_queue, &command)) {
            if (command == CMD_PROCESS_IMAGE) {
                // Get a lock on the shared memory
                uint32_t save = spin_lock_blocking(memory_lock);
                
                // Process the captured image
                bool process_success = process_image_for_inference(
                    g_capture_buffer, g_capture_size, g_process_buffer);
                
                // Release the lock
                spin_unlock(memory_lock, save);
                
                if (!process_success) {
                    printf("Core 1: Image processing failed\n");
                    uint32_t error_cmd = CMD_ERROR;
                    queue_try_add(&core1_to_core0_queue, &error_cmd);
                    continue;
                }
                
                // Fill input tensor with processed image
                fill_input_tensor(g_process_buffer);
                
                // Run inference with timing
                absolute_time_t inference_start = get_absolute_time();
                
                TfLiteStatus invoke_status = interpreter->Invoke();
                
                uint32_t inference_time_ms = absolute_time_diff_us(
                    inference_start, get_absolute_time()) / 1000;
                
                printf("Core 1: Inference took %ld ms\n", inference_time_ms);
                
                if (invoke_status != kTfLiteOk) {
                    printf("Core 1: Inference failed with status: %d\n", invoke_status);
                    uint32_t error_cmd = CMD_ERROR;
                    queue_try_add(&core1_to_core0_queue, &error_cmd);
                    continue;
                }
                
                // Process results and store in shared memory
                save = spin_lock_blocking(memory_lock);
                
                // Get lock on shared inference result
                InferenceResult* result = (InferenceResult*)&g_inference_result;
                result->inference_time_ms = inference_time_ms;
                
                // Process results based on output tensor type
                if (output->type == kTfLiteUInt8) {
                    uint8_t *results = output->data.uint8;
                    
                    for (int i = 0; i < 7; i++) {
                        float score = results[i] / 255.0f;  // Normalize to [0,1]
                        result->scores[i] = score;
                        // Apply thresholds
                        result->predictions[i] = (score > g_improved_thresholds[i]) ? 1 : 0;
                    }
                }
                else if (output->type == kTfLiteInt8) {
                    // For int8 output, we need to dequantize manually
                    int8_t *int8_results = output->data.int8;
                    float scale = output->params.scale;
                    int zero_point = output->params.zero_point;
                    
                    for (int i = 0; i < 7; i++) {
                        // Properly dequantize
                        float score = scale * (int8_results[i] - zero_point);
                        // Clip to 0-1 range
                        score = score < 0.0f ? 0.0f : (score > 1.0f ? 1.0f : score);
                        result->scores[i] = score;
                        // Apply thresholds
                        result->predictions[i] = (score > g_improved_thresholds[i]) ? 1 : 0;
                    }
                }
                else if (output->type == kTfLiteFloat32) {
                    float *float_results = output->data.f;
                    
                    for (int i = 0; i < 7; i++) {
                        float score = float_results[i];
                        // Apply sigmoid if needed
                        if (score < 0.0f || score > 1.0f) {
                            score = 1.0f / (1.0f + expf(-score));
                        }
                        result->scores[i] = score;
                        // Apply thresholds
                        result->predictions[i] = (score > g_improved_thresholds[i]) ? 1 : 0;
                    }
                }
                
                result->valid = true;
                spin_unlock(memory_lock, save);
                
                // Signal that inference is complete
                uint32_t complete_cmd = CMD_INFERENCE_COMPLETE;
                queue_try_add(&core1_to_core0_queue, &complete_cmd);
            }
        }
        
        // Small sleep to avoid tight loop
        sleep_us(100);
    }
}

// Main function - runs on Core 0
int main() {
    // Set inference result as invalid initially
    g_inference_result.valid = false;

    // Initialize all components
    if (!setup_hardware()) {
        printf("Hardware setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    if (!setup_display()) {
        printf("Display setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    if (!setup_camera()) {
        printf("Camera setup failed\n");
        while (true)
            sleep_ms(1000); // Halt
    }

    // Display status
    display_message("LOADING MODEL", "Please wait...", nullptr);
    
    // Launch Core 1 for ML processing
    multicore_launch_core1(core1_entry);
    
    // Wait for Core 1 to initialize TFLite
    printf("Core 0: Waiting for TensorFlow Lite initialization...\n");
    
    uint32_t response;
    bool init_success = false;
    absolute_time_t timeout = make_timeout_time_ms(10000);  // 10 second timeout
    
    while (!init_success) {
        if (queue_try_remove(&core1_to_core0_queue, &response)) {
            if (response == CMD_INFERENCE_COMPLETE) {
                init_success = true;
            }
            else if (response == CMD_ERROR) {
                printf("Core 0: Error during TensorFlow initialization\n");
                display_message("MODEL ERROR", "Restart device", nullptr);
                while (true)
                    sleep_ms(1000); // Halt
            }
        }
        
        // Check for timeout
        if (absolute_time_diff_us(get_absolute_time(), timeout) <= 0) {
            printf("Core 0: Timeout waiting for TensorFlow initialization\n");
            display_message("INIT TIMEOUT", "Restart device", nullptr);
            while (true)
                sleep_ms(1000); // Halt
        }
        
        sleep_ms(10);
    }

    // Show ready message
    display_message("SYSTEM READY", "Starting...", nullptr);
    sleep_ms(1000);

    // Initialize with a first result message
    display_message("AWAITING FIRST", "DETECTION", nullptr);
    
    // Flag to track if we need to update display
    bool display_needs_update = false;
    
    // Main processing loop
    while (true) {
        // Don't update display to "CAPTURING..." - keep previous results visible
        // Instead just log to console
        printf("Core 0: Capturing new image...\n");
        
        // Capture image
        uint32_t capture_size = 0;
        bool capture_success = capture_image_to_buffer(
            g_capture_buffer, sizeof(g_capture_buffer), &capture_size);
            
        if (!capture_success) {
            printf("Core 0: Image capture failed\n");
            // Only update display in case of error
            display_message("CAPTURE ERROR", "Retrying...", nullptr);
            sleep_ms(1000);
            continue;
        }
        
        // Set the capture size in the shared memory
        uint32_t save = spin_lock_blocking(memory_lock);
        g_capture_size = capture_size;
        spin_unlock(memory_lock, save);
        
        // Don't update display to "PROCESSING..." - keep previous results visible
        // Instead just log to console
        printf("Core 0: Processing image...\n");
        
        // Signal Core 1 to process the image
        uint32_t process_cmd = CMD_PROCESS_IMAGE;
        queue_try_add(&core0_to_core1_queue, &process_cmd);
        
        // Wait for Core 1 to complete processing
        bool processing_complete = false;
        bool processing_error = false;
        absolute_time_t timeout = make_timeout_time_ms(5000);  // 5 second timeout
        
        while (!processing_complete && !processing_error) {
            // Check for response from Core 1
            uint32_t response;
            if (queue_try_remove(&core1_to_core0_queue, &response)) {
                if (response == CMD_INFERENCE_COMPLETE) {
                    processing_complete = true;
                    display_needs_update = true;  // Mark that we need to update display
                }
                else if (response == CMD_ERROR) {
                    processing_error = true;
                }
            }
            
            // Check for timeout
            if (absolute_time_diff_us(get_absolute_time(), timeout) <= 0) {
                printf("Core 0: Timeout waiting for inference\n");
                display_message("INFERENCE TIMEOUT", "Retrying...", nullptr);
                processing_error = true;
            }
            
            sleep_ms(10);
        }
        
        if (processing_error) {
            sleep_ms(1000);
            continue;
        }
        
        // Only update the display when we have new results
        if (display_needs_update) {
            // Display results
            save = spin_lock_blocking(memory_lock);
            InferenceResult local_result;
            memcpy(&local_result, (void*)&g_inference_result, sizeof(InferenceResult));
            spin_unlock(memory_lock, save);
            
            display_results(&local_result);
            display_needs_update = false;  // Reset flag
        }
        
        // Small delay before next capture to avoid CPU hogging
        // but not too long to keep detection responsive
        sleep_ms(100);
    }

    return 0;
}   