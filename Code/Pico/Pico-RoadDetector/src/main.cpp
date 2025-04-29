/**
 * Pico-RoadDetector: Main Application
 * 
 * This application uses a Raspberry Pi Pico to run TensorFlow Lite machine learning
 * models for road object detection. It captures images from an ArduCAM camera,
 * processes them through a neural network, and displays results on an OLED screen.
 */

 #include "pico/stdlib.h"
 #include "pico/multicore.h"
 #include "hardware/i2c.h"
 #include "hardware/spi.h"
 #include "hardware/gpio.h"
 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
 
 // Project headers
 #include "hardware_config.h"
 #include "ArduCAM.h"
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
 
 // Global variables
 ArduCAM myCAM(OV5642, PIN_CS);
 uint8_t image_buffer[MODEL_WIDTH * MODEL_HEIGHT * 3]; // RGB buffer for model input
 
 // TensorFlow Lite globals
 namespace {
     // TF Lite error reporter
     tflite::MicroErrorReporter micro_error_reporter;
     tflite::ErrorReporter* error_reporter = &micro_error_reporter;
     
     // Model and interpreter
     const tflite::Model* model = nullptr;
     tflite::MicroInterpreter* interpreter = nullptr;
     
     // Input and output tensors
     TfLiteTensor* input = nullptr;
     TfLiteTensor* output = nullptr;
     
     // Arena for tensor memory allocation
     constexpr int kTensorArenaSize = 160 * 1024;
     alignas(16) uint8_t tensor_arena[kTensorArenaSize];
 }
 
 // Function prototypes
 void debug_print(const char* msg);
 void setup_hardware();
 void setup_camera();
 void setup_display();
 void setup_tensorflow();
 bool capture_and_process_image();
 void display_results(const uint8_t* results, int num_classes);
 void run_inference();
 bool test_camera_capture();
 void check_camera_status();
 
 /**
  * Helper function for debug output
  */
 void debug_print(const char* msg) {
     #if DEBUG
         printf("%s\n", msg);
     #endif
 }
 
 /**
  * Initialize hardware components
  */
 void setup_hardware() {
     // Initialize stdio
     stdio_init_all();
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
     gpio_init(PIN_CS);
     gpio_set_dir(PIN_CS, GPIO_OUT);
     gpio_put(PIN_CS, 1);
     
     // Initialize I2C multiplexer
     if (!pca9546_init(I2C_PORT, PCA9546_ADDR)) {
         printf("Failed to initialize PCA9546 multiplexer\n");
     }
     
     debug_print("Hardware initialized");
 }
 
 /**
  * Initialize OLED display
  */
 void setup_display() {
     debug_print("Setting up OLED display...");
     
     // Select OLED on multiplexer
     if (!pca9546_select(I2C_PORT, MUX_PORT_OLED)) {
         printf("Failed to select OLED on multiplexer\n");
     }
     
     // Initialize display
     ssd1306_init(I2C_PORT, SSD1306_ADDR);
     ssd1306_clear();
     ssd1306_draw_string(0, 0, "INITIALIZING...", 1);
     ssd1306_show();
     
     debug_print("OLED display ready");
 }

 bool verify_arducam_spi() {
    printf("Verifying ArduCAM SPI communication...\n");
    
    // Read ArduChip version register
    uint8_t version = myCAM.read_reg(0x40);
    printf("ArduChip version: 0x%02X (expected 0x40 for ArduCAM v2)\n", version);
    
    // Test write/read to a test register
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    uint8_t read_val = myCAM.read_reg(ARDUCHIP_TEST1);
    printf("Write 0x55, read: 0x%02X\n", read_val);
    
    myCAM.write_reg(ARDUCHIP_TEST1, 0xAA);
    read_val = myCAM.read_reg(ARDUCHIP_TEST1);
    printf("Write 0xAA, read: 0x%02X\n", read_val);
    
    // If we can read/write test register, SPI is working
    return (read_val == 0xAA);
}

/**
 * Check camera status by reading key registers
 */
void check_camera_status() {
    uint8_t reg_val;
    
    // Read some key registers
    myCAM.rdSensorReg16_8(0x3818, &reg_val);
    printf("Reg 0x3818 = 0x%02X\n", reg_val);
    
    myCAM.rdSensorReg16_8(0x300E, &reg_val);
    printf("Reg 0x300E = 0x%02X\n", reg_val);
    
    // Check if the camera is in JPEG mode
    myCAM.rdSensorReg16_8(0x4300, &reg_val);
    printf("Format control reg (0x4300) = 0x%02X (0x18 for JPEG)\n", reg_val);
}

/**
 * Test camera capture to verify functionality
 * 
 * @return true if successful, false otherwise
 */
bool test_camera_capture() {
    printf("Testing camera capture...\n");
    
    // Force JPEG mode for test capture
    myCAM.wrSensorReg16_8(0x4300, 0x18);
    sleep_ms(10);
    
    // Verify format register
    uint8_t format_reg;
    myCAM.rdSensorReg16_8(0x4300, &format_reg);
    printf("Format control before test: 0x%02X (should be 0x18)\n", format_reg);
    
    if (format_reg != 0x18) {
        printf("ERROR: Camera not in JPEG mode\n");
        return false;
    }
    
    // Reset FIFO before capture
    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();
    sleep_ms(50);
    
    // Start capture
    myCAM.start_capture();
    printf("Test capture started\n");
    
    // Wait for capture with timeout
    bool capture_done = false;
    for (int i = 0; i < 300; i++) {
        if (myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
            capture_done = true;
            printf("Test capture completed in %d ms\n", i * 10);
            break;
        }
        sleep_ms(10);
    }
    
    if (!capture_done) {
        printf("Test capture timed out\n");
        return false;
    }
    
    // Read FIFO length
    uint32_t length = myCAM.read_fifo_length();
    printf("Test image size: %d bytes\n", length);
    
    // Check if image size is reasonable
    if (length < 1000 || length > 200000) {
        printf("Test image size is unreasonable: %d bytes\n", length);
        return false;
    }
    
    // Allocate buffer for the entire image (or a large chunk of it for testing)
    const size_t MAX_BUFFER = 8192; // Use a reasonable size for testing
    uint8_t* buffer = (uint8_t*)malloc(MIN(length, MAX_BUFFER));
    
    if (!buffer) {
        printf("Failed to allocate buffer for image\n");
        return false;
    }
    
    // Read image data from FIFO
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();
    
    // Read JPEG header first to confirm it's valid
    for (int i = 0; i < 32 && i < length; i++) {
        spi_read_blocking(SPI_PORT, 0, &buffer[i], 1);
    }
    
    // For testing, we can read more data (up to our buffer limit)
    for (int i = 32; i < MIN(length, MAX_BUFFER); i++) {
        spi_read_blocking(SPI_PORT, 0, &buffer[i], 1);
    }
    
    myCAM.CS_HIGH();
    
    // Print first bytes for debugging
    printf("JPEG header: ");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", buffer[i]);
    }
    printf("\n");
    
    // Check for JPEG signature (0xFF 0xD8)
    bool valid_jpeg = (buffer[0] == 0xFF && buffer[1] == 0xD8);
    
    // Cleanup
    free(buffer);
    
    if (!valid_jpeg) {
        printf("Invalid JPEG header\n");
        return false;
    }
    
    printf("Valid JPEG data captured\n");
    return true;
}
 
bool direct_init_camera() {
    printf("Performing direct camera initialization...\n");
    
    // Step 1: Software reset
    myCAM.wrSensorReg16_8(0x3008, 0x80);
    sleep_ms(100);
    
    // Step 2: Set system settings
    myCAM.wrSensorReg16_8(0x3008, 0x00); // System control - no reset, no standby
    myCAM.wrSensorReg16_8(0x3017, 0xff); // Frex, Vsync, Href, PCLK, D[9:6] I/O Control
    myCAM.wrSensorReg16_8(0x3018, 0xff); // D[5:0], GPIO[1:0] I/O Control
    
    // Step 3: Set resolution registers for QVGA (320x240)
    myCAM.wrSensorReg16_8(0x3800, 0x00); // HREF start high byte
    myCAM.wrSensorReg16_8(0x3801, 0x00); // HREF start low byte
    myCAM.wrSensorReg16_8(0x3802, 0x00); // VREF start high byte
    myCAM.wrSensorReg16_8(0x3803, 0x00); // VREF start low byte
    myCAM.wrSensorReg16_8(0x3804, 0x0a); // HREF width high byte
    myCAM.wrSensorReg16_8(0x3805, 0x3f); // HREF width low byte
    myCAM.wrSensorReg16_8(0x3806, 0x07); // VREF height high byte
    myCAM.wrSensorReg16_8(0x3807, 0x9f); // VREF height low byte
    myCAM.wrSensorReg16_8(0x3808, 0x01); // Output width high byte (320)
    myCAM.wrSensorReg16_8(0x3809, 0x40); // Output width low byte
    myCAM.wrSensorReg16_8(0x380a, 0x00); // Output height high byte (240)
    myCAM.wrSensorReg16_8(0x380b, 0xF0); // Output height low byte
    
    // Step 4: Set critical JPEG registers
    myCAM.wrSensorReg16_8(0x4300, 0x18); // Format control - YUV422, JPEG
    myCAM.wrSensorReg16_8(0x4740, 0x21); // SLEW rate control for stable JPEG
    myCAM.wrSensorReg16_8(0x501f, 0x00); // ISP Format (YUV422)
    myCAM.wrSensorReg16_8(0x4713, 0x03); // JPEG mode 3
    myCAM.wrSensorReg16_8(0x4407, 0x04); // Quantization scale
    myCAM.wrSensorReg16_8(0x460b, 0x35); // VFIFO control
    myCAM.wrSensorReg16_8(0x460c, 0x22); // VFIFO control
    
    // Step 5: Set JPEG specific setup
    myCAM.wrSensorReg16_8(0x3818, 0xa8); // Timing control, compression
    myCAM.wrSensorReg16_8(0x3621, 0x10); // Array control
    myCAM.wrSensorReg16_8(0x3801, 0xb0); // Timing HS
    
    // Verify settings
    uint8_t format_reg;
    myCAM.rdSensorReg16_8(0x4300, &format_reg);
    printf("Format control reg after direct init: 0x%02X\n", format_reg);
    
    // Step 6: Reset FIFO
    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();
    
    return (format_reg == 0x18);
}

bool test_alternative_fifo_read() {
    printf("Testing alternative FIFO read method...\n");
    
    // Prepare FIFO for capture
    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();
    
    // Explicitly toggle CS to ensure ArduChip wakes up
    gpio_put(PIN_CS, 1);
    sleep_ms(5);
    gpio_put(PIN_CS, 0);
    sleep_ms(5);
    gpio_put(PIN_CS, 1);
    sleep_ms(5);
    
    // Start capture with clear CS toggling
    myCAM.CS_LOW();
    uint8_t cmd = ARDUCHIP_FIFO | FIFO_START_MASK;
    spi_write_blocking(SPI_PORT, &cmd, 1);
    myCAM.CS_HIGH();
    
    printf("Waiting for capture to complete...\n");
    
    // Wait for capture with timeout
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    bool capture_done = false;
    
    while (!capture_done) {
        myCAM.CS_LOW();
        uint8_t reg = ARDUCHIP_TRIG;
        spi_write_blocking(SPI_PORT, &reg, 1);
        uint8_t status;
        spi_read_blocking(SPI_PORT, 0, &status, 1);
        myCAM.CS_HIGH();
        
        if (status & CAP_DONE_MASK) {
            capture_done = true;
            printf("Capture completed\n");
        }
        
        if (to_ms_since_boot(get_absolute_time()) - start_time > 3000) {
            printf("Capture timed out after 3 seconds\n");
            return false;
        }
        
        sleep_ms(10);
    }
    
    // Read FIFO length with explicit CS control
    uint32_t length = 0;
    
    // Read SIZE1
    myCAM.CS_LOW();
    uint8_t reg = FIFO_SIZE1;
    spi_write_blocking(SPI_PORT, &reg, 1);
    uint8_t size1;
    spi_read_blocking(SPI_PORT, 0, &size1, 1);
    myCAM.CS_HIGH();
    
    // Read SIZE2
    myCAM.CS_LOW();
    reg = FIFO_SIZE2;
    spi_write_blocking(SPI_PORT, &reg, 1);
    uint8_t size2;
    spi_read_blocking(SPI_PORT, 0, &size2, 1);
    myCAM.CS_HIGH();
    
    // Read SIZE3
    myCAM.CS_LOW();
    reg = FIFO_SIZE3;
    spi_write_blocking(SPI_PORT, &reg, 1);
    uint8_t size3;
    spi_read_blocking(SPI_PORT, 0, &size3, 1);
    myCAM.CS_HIGH();
    
    // Calculate full length
    length = ((size3 & 0x7f) << 16) | (size2 << 8) | size1;
    printf("FIFO Length: %d bytes\n", length);
    
    if (length <= 20) {
        printf("Image too small, not a valid JPEG\n");
        return false;
    }
    
    // Read the first few bytes to check for JPEG header
    uint8_t header[32];
    
    myCAM.CS_LOW();
    reg = BURST_FIFO_READ;
    spi_write_blocking(SPI_PORT, &reg, 1);
    
    for (int i = 0; i < 32 && i < length; i++) {
        spi_read_blocking(SPI_PORT, 0, &header[i], 1);
    }
    
    myCAM.CS_HIGH();
    
    printf("First 16 bytes: ");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", header[i]);
    }
    printf("\n");
    
    // Check for JPEG header (0xFF 0xD8)
    if (header[0] == 0xFF && header[1] == 0xD8) {
        printf("Valid JPEG header detected!\n");
        return true;
    } else {
        printf("Invalid image format - no JPEG header\n");
        return false;
    }
}

bool test_direct_fifo_capture() {
    printf("Testing direct FIFO capture with explicit control...\n");
    
    // 1. Reset FIFO and all flags with more thorough approach
    myCAM.write_reg(ARDUCHIP_FIFO, FIFO_CLEAR_MASK);
    sleep_ms(20);
    myCAM.write_reg(ARDUCHIP_FIFO, FIFO_RDPTR_RST_MASK);
    sleep_ms(20);
    myCAM.write_reg(ARDUCHIP_FIFO, FIFO_WRPTR_RST_MASK);
    sleep_ms(20);
    myCAM.clear_fifo_flag();
    sleep_ms(20);
    
    // 2. Verify reset was successful
    uint32_t length_after_reset = myCAM.read_fifo_length();
    printf("FIFO length after reset: %d bytes (should be 0)\n", length_after_reset);
    
    if (length_after_reset != 0) {
        printf("ERROR: FIFO reset failed!\n");
        return false;
    }
    
    // 3. Set camera mode to JPEG and check settings
    myCAM.wrSensorReg16_8(0x4300, 0x18);  // Format control - YUV422 + JPEG
    myCAM.wrSensorReg16_8(0x501F, 0x00);  // ISP output format control
    myCAM.wrSensorReg16_8(0x4713, 0x03);  // JPEG mode 3
    sleep_ms(20);
    
    // 4. Toggle the SPI CS line manually
    gpio_put(PIN_CS, 1);
    sleep_ms(10);
    gpio_put(PIN_CS, 0);
    sleep_ms(10);
    gpio_put(PIN_CS, 1);
    sleep_ms(10);
    
    // 5. Start capture with manual command writing
    myCAM.CS_LOW();
    uint8_t cmd = ARDUCHIP_FIFO | FIFO_START_MASK;
    spi_write_blocking(SPI_PORT, &cmd, 1);
    myCAM.CS_HIGH();
    sleep_ms(10);
    
    printf("Manual capture started...\n");
    
    // 6. Wait for capture completion
    bool capture_done = false;
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    
    while (!capture_done) {
        // Read directly with manual CS control
        myCAM.CS_LOW();
        uint8_t trig_addr = ARDUCHIP_TRIG;
        spi_write_blocking(SPI_PORT, &trig_addr, 1);
        uint8_t trig_val;
        spi_read_blocking(SPI_PORT, 0, &trig_val, 1);
        myCAM.CS_HIGH();
        
        printf("Trigger status: 0x%02X\n", trig_val);
        
        if (trig_val & CAP_DONE_MASK) {
            capture_done = true;
            printf("Manual capture complete\n");
        }
        
        if (to_ms_since_boot(get_absolute_time()) - start_time > 5000) {
            printf("Manual capture timed out\n");
            return false;
        }
        
        sleep_ms(100);
    }
    
    // 7. Read FIFO length with entirely manual method
    uint32_t length = 0;
    
    // Read SIZE1
    myCAM.CS_LOW();
    uint8_t size1_addr = FIFO_SIZE1;
    spi_write_blocking(SPI_PORT, &size1_addr, 1);
    uint8_t size1;
    spi_read_blocking(SPI_PORT, 0, &size1, 1);
    myCAM.CS_HIGH();
    sleep_ms(5);
    
    // Read SIZE2
    myCAM.CS_LOW();
    uint8_t size2_addr = FIFO_SIZE2;
    spi_write_blocking(SPI_PORT, &size2_addr, 1);
    uint8_t size2;
    spi_read_blocking(SPI_PORT, 0, &size2, 1);
    myCAM.CS_HIGH();
    sleep_ms(5);
    
    // Read SIZE3
    myCAM.CS_LOW();
    uint8_t size3_addr = FIFO_SIZE3;
    spi_write_blocking(SPI_PORT, &size3_addr, 1);
    uint8_t size3;
    spi_read_blocking(SPI_PORT, 0, &size3, 1);
    myCAM.CS_HIGH();
    sleep_ms(5);
    
    // Calculate full length
    length = ((size3 & 0x7f) << 16) | (size2 << 8) | size1;
    printf("Manual read - FIFO length: %d bytes\n", length);
    
    // Print individual size bytes for debugging
    printf("Size bytes: %02X %02X %02X\n", size1, size2, size3);
    
    if (length <= 20) {
        printf("Image too small, not a valid JPEG\n");
        return false;
    }
    
    // 8. Read first bytes of data to verify JPEG format
    myCAM.CS_LOW();
    uint8_t burst_addr = BURST_FIFO_READ;
    spi_write_blocking(SPI_PORT, &burst_addr, 1);
    
    uint8_t header[32];
    for (int i = 0; i < 32 && i < length; i++) {
        spi_read_blocking(SPI_PORT, 0, &header[i], 1);
    }
    
    myCAM.CS_HIGH();
    
    printf("Image header bytes: ");
    for (int i = 0; i < 16; i++) {
        printf("%02X ", header[i]);
    }
    printf("\n");
    
    // 9. Check for JPEG header
    if (header[0] == 0xFF && header[1] == 0xD8) {
        printf("Valid JPEG header detected\n");
        return true;
    } else {
        printf("Invalid image header (not JPEG)\n");
        return false;
    }
}

void reconfigure_arducam_controller() {
    printf("Reconfiguring ArduCAM controller with custom settings...\n");
    
    // Full ArduChip configuration
    myCAM.write_reg(ARDUCHIP_MODE, 0x00);  // Reset to default mode
    sleep_ms(10);
    myCAM.write_reg(ARDUCHIP_MODE, 0x01);  // Set to CAM2LCD mode
    sleep_ms(10);
    
    // Configure FIFO control
    myCAM.write_reg(ARDUCHIP_FIFO, 0x02);  // VSYNC level active low
    sleep_ms(10);
    
    // Set timing control
    myCAM.write_reg(ARDUCHIP_TIM, 0x00);  // Default timing
    sleep_ms(10);
    
    // Configure GPIO
    myCAM.write_reg(ARDUCHIP_GPIO, 0x01);  // Set GPIO0 to output high
    sleep_ms(10);
    
    // Reset all FIFO flags
    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();
    
    printf("ArduCAM controller reconfigured\n");
}

 /**
  * Initialize ArduCAM camera
  */
 void setup_camera() {
    debug_print("Setting up ArduCAM...");
    reconfigure_arducam_controller();
    // Clear retry counter
    int retry_count = 0;
    bool camera_ready = false;
    
    while (!camera_ready && retry_count < 5) {
        printf("Camera setup attempt #%d\n", retry_count + 1);
        
        if (!pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM)) {
            printf("ERROR: Failed to select camera channel on multiplexer\n");
            retry_count++;
            sleep_ms(100);
            continue;
        }
        
        // Verify SPI communication first
        if (!verify_arducam_spi()) {
            printf("ERROR: ArduCAM SPI communication failed!\n");
            retry_count++;
            sleep_ms(100);
            continue;
        }

        if (!camera_ready) {
            printf("Trying direct FIFO capture method as last resort...\n");
            if (test_direct_fifo_capture()) {
                printf("Direct FIFO capture method works! Using this approach.\n");
                camera_ready = true;
            } else {
                printf("CRITICAL ERROR: All camera capture methods failed\n");
            }
        }
        
        // Hard reset the camera
        myCAM.wrSensorReg16_8(0x3008, 0x80);
        sleep_ms(100);
        
        // Check camera ID
        uint8_t vid, pid;
        myCAM.wrSensorReg16_8(0xff, 0x01);
        myCAM.rdSensorReg16_8(0x300A, &vid);
        myCAM.rdSensorReg16_8(0x300B, &pid);
        printf("Camera ID check - VID: 0x%02X, PID: 0x%02X\n", vid, pid);
        
        if (vid != 0x56 || pid != 0x42) {
            printf("ERROR: Camera ID mismatch!\n");
            retry_count++;
            sleep_ms(100);
            continue;
        }
        
        // Full initialization
        myCAM.set_format(JPEG);
        myCAM.InitCAM();
        myCAM.OV5642_set_JPEG_size(OV5642_320x240);
        sleep_ms(100);
        
        // Set critical JPEG registers
        myCAM.wrSensorReg16_8(0x4300, 0x18);
        myCAM.wrSensorReg16_8(0x3818, 0xA8);
        myCAM.wrSensorReg16_8(0x3621, 0x10);
        myCAM.wrSensorReg16_8(0x3801, 0xB0);
        myCAM.wrSensorReg16_8(0x4407, 0x04);
        myCAM.wrSensorReg16_8(0x300E, 0x45);
        sleep_ms(100);
        
        // Verify some key registers
        uint8_t reg_value;
        myCAM.rdSensorReg16_8(0x4300, &reg_value);
        printf("Format control reg: 0x%02X (should be 0x18)\n", reg_value);
        
        // Before running normal test, try the alternative method
        printf("Testing alternative FIFO read method...\n");
        if (test_alternative_fifo_read()) {
            printf("Alternative FIFO test SUCCESSFUL!\n");
            camera_ready = true;
            break;
        } else {
            printf("Alternative FIFO test failed, trying standard test...\n");
            if (test_camera_capture()) {
                printf("Standard camera test SUCCESSFUL!\n");
                camera_ready = true;
                break;
            }
        }
        
        printf("Camera tests failed, retrying setup...\n");
        retry_count++;
        sleep_ms(100);
    }
    
    if (!camera_ready) {
        printf("CRITICAL ERROR: Failed to initialize camera after multiple attempts\n");
        // Consider showing error on display
    } else {
        printf("Camera successfully initialized!\n");
    }
}
 
 /**
  * Initialize TensorFlow Lite
  */
 void setup_tensorflow() {
    debug_print("Setting up TensorFlow Lite...");
    
    // Update display with status
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "LOADING MODEL", 1);
    ssd1306_show();
    
    // Print memory info
    printf("Tensor arena size: %d KB\n", kTensorArenaSize / 1024);
    
    // Map the model into a usable data structure
    printf("Getting model from model_data (%d bytes)...\n", model_data_len);
    model = tflite::GetModel(model_data);
    
    if (!model) {
        printf("ERROR: Failed to get model\n");
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "MODEL ERROR", 1);
        ssd1306_show();
        while (1) { sleep_ms(100); } // Halt
    }
    
    printf("Model version: %d, Schema version: %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema version mismatch!");
        
        // Report error on display
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "MODEL ERROR", 1);
        ssd1306_show();
        
        while (1) { sleep_ms(100); } // Halt
    }
    
    printf("Creating op resolver...\n");
    static tflite::MicroMutableOpResolver<20> resolver; // Increased from 10 to 15
    
    // Register operations (add all that might be needed by your model)
    printf("Registering operations...\n");
    
   // Basic operations for MobileNetV2
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();

    // Quantization operations
    resolver.AddDequantize();
    resolver.AddQuantize();

    // Additional operations
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddStridedSlice();
    resolver.AddLogistic(); // This was missing before!

    // Add these additional ops that might be needed
    resolver.AddMul();  // To address the arm_elementwise_mul_s8 error
    
    printf("Building interpreter...\n");
 
    
    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
       model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate memory for the model's tensors
    printf("Allocating tensors...\n");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    printf("Allocation status: %d (0 is success)\n", allocate_status);
    
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed with status %d", allocate_status);
        
        // Report error on display
        pca9546_select(I2C_PORT, MUX_PORT_OLED);
        ssd1306_clear();
        ssd1306_draw_string(0, 0, "TENSOR ERROR", 1);
        ssd1306_show();
        
        while (1) { sleep_ms(100); } // Halt
    }
    
    // Get pointers to input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Print tensor information for debugging
    printf("Input tensor type: %d (kTfLiteInt8=%d, kTfLiteUInt8=%d, kTfLiteFloat32=%d)\n", 
           input->type, kTfLiteInt8, kTfLiteUInt8, kTfLiteFloat32);
    printf("Output tensor type: %d (kTfLiteInt8=%d, kTfLiteUInt8=%d, kTfLiteFloat32=%d)\n", 
           output->type, kTfLiteInt8, kTfLiteUInt8, kTfLiteFloat32);
    printf("Input tensor dims: %d x %d x %d x %d\n", 
           input->dims->data[0], input->dims->data[1], 
           input->dims->data[2], input->dims->data[3]);
    printf("Output tensor dims: %d x %d\n", 
           output->dims->data[0], output->dims->data[1]);
    printf("Input tensor details - zero_point: %d, scale: %f\n", 
           input->params.zero_point, input->params.scale);
    printf("Output tensor details - zero_point: %d, scale: %f\n", 
           output->params.zero_point, output->params.scale);
    printf("Arena size: %d bytes used, %d bytes available\n",
           interpreter->arena_used_bytes(), kTensorArenaSize);
    
    // Display success
    pca9546_select(I2C_PORT, MUX_PORT_OLED);
    ssd1306_clear();
    ssd1306_draw_string(0, 0, "MODEL READY", 1);
    ssd1306_show();
    sleep_ms(500);
    
    debug_print("TensorFlow Lite ready");
}
 
 /**
  * Capture and process an image from the camera for inference
  * 
  * @return true if successful, false otherwise
  */
 bool capture_and_process_image() {
    debug_print("Capturing image...");
    
    // Switch to ArduCAM
    pca9546_select(I2C_PORT, MUX_PORT_ARDUCAM);
    printf("ArduCAM selected on multiplexer\n");
    
    // Force JPEG mode again before capture
  // Force all critical JPEG registers again
myCAM.wrSensorReg16_8(0x4300, 0x18); // Format: YUV422 + JPEG
myCAM.wrSensorReg16_8(0x501F, 0x00); // ISP output YUV422
myCAM.wrSensorReg16_8(0x3818, 0xA8); // Timing
myCAM.wrSensorReg16_8(0x3621, 0x10); // Array control
myCAM.wrSensorReg16_8(0x3801, 0xB0); // HSYNC position
myCAM.wrSensorReg16_8(0x4407, 0x04); // Quantization scale (optional)

sleep_ms(100); // Let the sensor apply the changes

    
    // Start capture
    myCAM.flush_fifo();
    sleep_ms(50);
    myCAM.clear_fifo_flag();
    sleep_ms(50);
    
    // Small extra SPI toggle to make sure ArduChip is awake
    gpio_put(PIN_CS, 1);
    sleep_ms(5);
    gpio_put(PIN_CS, 0);
    sleep_ms(5);
    gpio_put(PIN_CS, 1);
    sleep_ms(5);
    // 🔥 Re-force critical JPEG and compression registers before capture
myCAM.wrSensorReg16_8(0x4300, 0x18); // Format Control - YUV422 + JPEG
myCAM.wrSensorReg16_8(0x501F, 0x00); // ISP output format control
myCAM.wrSensorReg16_8(0x3818, 0xA8); // Timing for compression
myCAM.wrSensorReg16_8(0x3621, 0x10); // Array control
myCAM.wrSensorReg16_8(0x3801, 0xB0); // HSYNC control
myCAM.wrSensorReg16_8(0x4407, 0x04); // JPEG quantization scale
sleep_ms(100);

    printf("Starting capture...\n");
    myCAM.start_capture();
    
    // Wait for capture to complete
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    bool capture_timeout = false;
    
    while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        if (to_ms_since_boot(get_absolute_time()) - start_time > 3000) {
            capture_timeout = true;
            break;
        }
        sleep_ms(10);
    }
    
    if (capture_timeout) {
        printf("Error: Camera capture timeout\n");
        return false;
    }
    
    printf("Capture completed successfully\n");
    
    // Read captured data
    uint32_t length = myCAM.read_fifo_length();
    printf("FIFO length: %d bytes\n", length);
    
    if (length > 256000 || length < 20) {
        printf("Error: Image size invalid: %d bytes\n", length);
        return false;
    }
    
    // Allocate memory for entire RAW data
    uint8_t* raw_buffer = (uint8_t*)malloc(length);
    if (!raw_buffer) {
        printf("Error: Failed to allocate memory for RAW data\n");
        return false;
    }
    
    // Read all data in chunks
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();
    
    // Read data in chunks (e.g., 1024 bytes at a time) for better stability
    const size_t CHUNK_SIZE = 1024;
    for (uint32_t i = 0; i < length; i += CHUNK_SIZE) {
        size_t bytes_to_read = MIN(CHUNK_SIZE, length - i);
        for (size_t j = 0; j < bytes_to_read; j++) {
            spi_read_blocking(SPI_PORT, 0, &raw_buffer[i + j], 1);
        }
    }
    
    myCAM.CS_HIGH();
    
    // Print first few bytes for debugging
    printf("First 16 bytes: ");
    for (int i = 0; i < 16 && i < length; i++) {
        printf("%02X ", raw_buffer[i]);
    }
    printf("\n");
    
    // Create grayscale image from raw data
    uint8_t* gray_img = (uint8_t*)malloc(320 * 240);
    if (!gray_img) {
        free(raw_buffer);
        printf("Error: Failed to allocate grayscale buffer\n");
        return false;
    }
    
    // Simple conversion from raw data to grayscale
    for (int i = 0; i < 320 * 240; i++) {
        uint32_t idx = (i * length) / (320 * 240);
        idx = idx < length ? idx : length - 1;
        gray_img[i] = raw_buffer[idx];
    }

    // Check tensor type and handle accordingly
    printf("Input tensor type: %d (UInt8=%d, Float32=%d, Int8=%d)\n", 
           input->type, kTfLiteUInt8, kTfLiteFloat32, kTfLiteInt8);
    
    if (input->type == kTfLiteInt8) {
        // For int8 quantized model input
        int8_t* input_data = input->data.int8;
        
        // If the scale is zero (which happens in your case), use a default scale
        float scale = (input->params.scale <= 0.0001f) ? 0.007874f : input->params.scale;
        int zero_point = input->params.zero_point;
        
        printf("Using scale=%f and zero_point=%d for INT8 tensors\n", scale, zero_point);
        
        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Calculate source coordinate in original image
                int src_x = x * 320 / MODEL_WIDTH;
                int src_y = y * 240 / MODEL_HEIGHT;
                int src_idx = src_y * 320 + src_x;
                
                // Get pixel value
                uint8_t pixel = gray_img[src_idx];
                
                // Simple conversion to INT8 range
                // Map from 0-255 to -128 to 127
                int8_t quantized = (int8_t)(pixel - 128);
                
                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = quantized;  // R
                    input_data[dst_idx + 1] = quantized;  // G
                    input_data[dst_idx + 2] = quantized;  // B
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
        uint8_t* input_data = input->data.uint8;
        
        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Calculate source coordinate in original image
                int src_x = x * 320 / MODEL_WIDTH;
                int src_y = y * 240 / MODEL_HEIGHT;
                int src_idx = src_y * 320 + src_x;
                
                // Get pixel value
                uint8_t pixel = gray_img[src_idx];
                
                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel;  // R
                    input_data[dst_idx + 1] = pixel;  // G
                    input_data[dst_idx + 2] = pixel;  // B
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
        float* input_data = input->data.f;
        
        for (int y = 0; y < MODEL_HEIGHT; y++) {
            for (int x = 0; x < MODEL_WIDTH; x++) {
                // Calculate source coordinate in original image
                int src_x = x * 320 / MODEL_WIDTH;
                int src_y = y * 240 / MODEL_HEIGHT;
                int src_idx = src_y * 320 + src_x;
                
                // Get pixel value and normalize to 0-1 range
                float pixel = gray_img[src_idx] / 255.0f;
                
                // For RGB model inputs with 3 channels
                if (input->dims->data[3] == 3) {
                    int dst_idx = (y * MODEL_WIDTH + x) * 3;
                    input_data[dst_idx + 0] = pixel;  // R
                    input_data[dst_idx + 1] = pixel;  // G
                    input_data[dst_idx + 2] = pixel;  // B
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
        free(gray_img);
        free(raw_buffer);
        return false;
    }
    
    // Print debug info about input tensor
    printf("Input tensor details - zeropt: %d, scale: %f\n", 
           input->params.zero_point, input->params.scale);
    
    // Clean up
    free(gray_img);
    free(raw_buffer);
    
    return true;
}
    
    // Rest of function remains the same for tensor processing
    // ...
 /**
  * Display inference results on the OLED
  * 
  * @param results pointer to the inference results
  * @param num_classes number of classes in the results
  */
 void display_results(const uint8_t* results, int num_classes) {
     // Find the class with highest score
     int max_score = 0;
     int max_index = 0;
     
     for (int i = 0; i < num_classes; i++) {
         if (results[i] > max_score) {
             max_score = results[i];
             max_index = i;
         }
     }
     
     // Limit to the number of classes we know about
     if (max_index >= sizeof(class_names) / sizeof(class_names[0])) {
         max_index = 0;
     }
     
     // Calculate confidence percentage
     int confidence = (max_score * 100) / 255;
     
     // Display results on OLED
     pca9546_select(I2C_PORT, MUX_PORT_OLED);
     ssd1306_clear();
     
     // Display class name
     ssd1306_draw_string(0, 0, class_names[max_index], 1);
     
     // Show confidence score
     char score_text[16];
     snprintf(score_text, sizeof(score_text), "CONF: %d%%", confidence);
     ssd1306_draw_string(0, 16, score_text, 1);
     
     ssd1306_show();
     
     // Print to console as well
     printf("Detected: %s (confidence: %d%%)\n", class_names[max_index], confidence);
 }
 
 /**
  * Run inference on the captured image
  */
 void run_inference() {
     // Update display
     pca9546_select(I2C_PORT, MUX_PORT_OLED);
     ssd1306_clear();
     ssd1306_draw_string(0, 0, "CAPTURING...", 1);
     ssd1306_show();
     
     // Capture and process image
     if (!capture_and_process_image()) {
         // Show error on display
         pca9546_select(I2C_PORT, MUX_PORT_OLED);
         ssd1306_clear();
         ssd1306_draw_string(0, 0, "CAPTURE ERROR", 1);
         ssd1306_show();
         sleep_ms(1000);
         return;
     }
     
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
     printf("Inference took %d ms\n", inference_time);
     
     if (invoke_status != kTfLiteOk) {
         printf("Inference failed with status: %d\n", invoke_status);
         
         // Show error on display
         pca9546_select(I2C_PORT, MUX_PORT_OLED);
         ssd1306_clear();
         ssd1306_draw_string(0, 0, "INFERENCE ERROR", 1);
         ssd1306_show();
         sleep_ms(1000);
         return;
     }
     
     // Get output tensor data and type
     printf("Output tensor type: %d\n", output->type);
     
     // Handle different output tensor types
     if (output->type == kTfLiteUInt8) {
         uint8_t* results = output->data.uint8;
         int num_classes = output->dims->data[1];
         display_results(results, num_classes);
     } 
     else if (output->type == kTfLiteInt8) {
         // Convert int8 to uint8 for display
         int8_t* int8_results = output->data.int8;
         int num_classes = output->dims->data[1];
         
         // Allocate temporary buffer
         uint8_t* uint8_results = (uint8_t*)malloc(num_classes);
         if (!uint8_results) {
             printf("Failed to allocate memory for results conversion\n");
             return;
         }
         
         // Convert from int8 to uint8
         for (int i = 0; i < num_classes; i++) {
             // Map from -128..127 to 0..255
             uint8_results[i] = (uint8_t)((int8_results[i] + 128) & 0xFF);
         }
         
         display_results(uint8_results, num_classes);
         free(uint8_results);
     }
     else {
         printf("Unsupported output tensor type: %d\n", output->type);
     }
 }
 
 /**
  * Main application entry point
  */
 int main() {
     // Initialize all components
     setup_hardware();
     setup_display();
     setup_camera();
     setup_tensorflow();
     
     // Additional check for camera and tensor type compatibility
     printf("Final camera and tensor compatibility check:\n");
     check_camera_status();
     printf("Input tensor type: %d, Output tensor type: %d\n", input->type, output->type);

     printf("Reinitializing camera fully after compatibility check...\n");
     setup_camera();
     sleep_ms(500);
     
     // Show ready message
     pca9546_select(I2C_PORT, MUX_PORT_OLED);
     ssd1306_clear();
     ssd1306_draw_string(0, 0, "SYSTEM READY", 1);
     ssd1306_show();
     sleep_ms(1000);
     
     // Main loop
     while (true) {
         run_inference();
         sleep_ms(2000);
     }
     
     return 0;
 }