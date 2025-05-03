#include <stdio.h>
#include "hardware_config.h"
#include "pca9546.h"

// Default configuration
const pca9546_config_t PCA9546_DEFAULT_CONFIG = {
    .timeout_ms = 100,
    .verify_selection = true,
    .retry_count = 2
};

// Current configuration (initialized to default at startup)
static pca9546_config_t current_config = {
    .timeout_ms = 100,
    .verify_selection = true,
    .retry_count = 2
};

// Error message strings corresponding to error codes
static const char* ERROR_STRINGS[] = {
    "Success",
    "Communication failed",
    "Invalid channel specified",
    "Channel verification failed",
    "Device is busy",
    "Operation timed out"
};

int pca9546_init(i2c_inst_t *i2c, uint8_t addr) {
    return pca9546_init_with_config(i2c, addr, &PCA9546_DEFAULT_CONFIG);
}

int pca9546_init_with_config(i2c_inst_t *i2c, uint8_t addr, const pca9546_config_t *config) {
    // Store configuration
    if (config != NULL) {
        current_config = *config;
    } else {
        current_config = PCA9546_DEFAULT_CONFIG;
    }
    
    // Check if the multiplexer is responsive
    uint8_t rxdata;
    int ret = i2c_read_blocking(i2c, addr, &rxdata, 1, false);
    
    if (ret < 0) {
        printf("PCA9546: Device not responding at address 0x%02X\n", addr);
        return PCA9546_ERROR_COMM_FAILED;
    }
    
    // Reset all channels to disabled state
    int result = pca9546_disable_all_channels(i2c, addr);
    
    if (result != PCA9546_SUCCESS) {
        printf("PCA9546: Failed to initialize device at address 0x%02X\n", addr);
        return result;
    }
    
    printf("PCA9546: Successfully initialized at address 0x%02X\n", addr);
    return PCA9546_SUCCESS;
}

int pca9546_select(i2c_inst_t *i2c, uint8_t port) {
    // Wrapper function to simplify channel selection by port number (0-3)
    if (port > 3) {
        printf("PCA9546: Invalid port number %d (must be 0-3)\n", port);
        return PCA9546_ERROR_INVALID_CHANNEL;
    }
    
    uint8_t channel_mask = 1 << port;
    return pca9546_select_channel(i2c, PCA9546_ADDR, channel_mask);
}

int pca9546_select_channel(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask) {
    // Check that a valid single channel bit is set (0, 1, 2, or 3)
    if (channel_mask != PCA9546_CHANNEL_0 && 
        channel_mask != PCA9546_CHANNEL_1 && 
        channel_mask != PCA9546_CHANNEL_2 && 
        channel_mask != PCA9546_CHANNEL_3) {
        printf("PCA9546: Invalid channel selection 0x%02X - must select a single channel\n", 
               channel_mask);
        return PCA9546_ERROR_INVALID_CHANNEL;
    }
    
    // Try operation up to retry_count times
    for (uint8_t retry = 0; retry <= current_config.retry_count; retry++) {
        // First disable all channels
        uint8_t disable_buf[1] = { 0 };
        int disable_ret = i2c_write_blocking(i2c, addr, disable_buf, 1, false);
        
        if (disable_ret < 0) {
            if (retry == current_config.retry_count) {
                printf("PCA9546: Failed to disable channels after %d retries\n", retry);
                return PCA9546_ERROR_COMM_FAILED;
            }
            sleep_ms(5);
            continue;
        }
        
        sleep_ms(5);  // Small delay for stability
        
        // Now enable the requested channel
        uint8_t buf[1] = { channel_mask };
        int ret = i2c_write_blocking(i2c, addr, buf, 1, false);
        
        if (ret < 0) {
            if (retry == current_config.retry_count) {
                printf("PCA9546: Failed to select channel 0x%02X with error %d\n", 
                       channel_mask, ret);
                return PCA9546_ERROR_COMM_FAILED;
            }
            sleep_ms(5);
            continue;
        }
        
        sleep_ms(5);  // Small delay for stability
        
        // Verify the channel was set correctly if configured to do so
        if (current_config.verify_selection) {
            uint8_t current_value;
            int read_ret = i2c_read_blocking(i2c, addr, &current_value, 1, false);
            
            if (read_ret < 0) {
                if (retry == current_config.retry_count) {
                    printf("PCA9546: Failed to verify channel selection\n");
                    return PCA9546_ERROR_VERIFY_FAILED;
                }
                sleep_ms(5);
                continue;
            }
            
            if (current_value != channel_mask) {
                printf("PCA9546: Channel verification failed - requested: 0x%02X, read: 0x%02X\n", 
                       channel_mask, current_value);
                
                if (retry == current_config.retry_count) {
                    return PCA9546_ERROR_VERIFY_FAILED;
                }
                
                // Try again with a longer delay
                sleep_ms(10);
                continue;
            }
        }
        
        // If we got here, operation was successful
        return PCA9546_SUCCESS;
    }
    
    // Should not reach here if retries are properly handled
    return PCA9546_ERROR_COMM_FAILED;
}

int pca9546_select_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask) {
    // Write the channel mask to the multiplexer
    uint8_t buf[1] = { channel_mask };
    
    for (uint8_t retry = 0; retry <= current_config.retry_count; retry++) {
        int ret = i2c_write_blocking(i2c, addr, buf, 1, false);
        
        if (ret < 0) {
            if (retry == current_config.retry_count) {
                printf("PCA9546: Failed to select channel mask 0x%02X with error %d\n", 
                       channel_mask, ret);
                return PCA9546_ERROR_COMM_FAILED;
            }
            sleep_ms(5);
            continue;
        }
        
        // Verify the channel was set correctly if configured to do so
        if (current_config.verify_selection) {
            uint8_t current_value;
            int verify_success = pca9546_get_selected_channels(i2c, addr, &current_value);
            
            if (verify_success != PCA9546_SUCCESS) {
                if (retry == current_config.retry_count) {
                    printf("PCA9546: Failed to verify channel selection\n");
                    return PCA9546_ERROR_VERIFY_FAILED;
                }
                sleep_ms(5);
                continue;
            }
            
            if (current_value != channel_mask) {
                printf("PCA9546: Channel verification failed - requested: 0x%02X, read: 0x%02X\n", 
                       channel_mask, current_value);
                
                if (retry == current_config.retry_count) {
                    return PCA9546_ERROR_VERIFY_FAILED;
                }
                
                sleep_ms(10);
                continue;
            }
        }
        
        return PCA9546_SUCCESS;
    }
    
    return PCA9546_ERROR_COMM_FAILED;
}

int pca9546_get_selected_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t *channel_mask) {
    if (channel_mask == NULL) {
        return PCA9546_ERROR_INVALID_CHANNEL;
    }
    
    for (uint8_t retry = 0; retry <= current_config.retry_count; retry++) {
        int ret = i2c_read_blocking(i2c, addr, channel_mask, 1, false);
        
        if (ret < 0) {
            if (retry == current_config.retry_count) {
                return PCA9546_ERROR_COMM_FAILED;
            }
            sleep_ms(5);
            continue;
        }
        
        return PCA9546_SUCCESS;
    }
    
    return PCA9546_ERROR_COMM_FAILED;
}

int pca9546_disable_all_channels(i2c_inst_t *i2c, uint8_t addr) {
    return pca9546_select_channels(i2c, addr, PCA9546_CHANNEL_NONE);
}

int pca9546_scan_channel(i2c_inst_t *i2c, uint8_t mux_addr, uint8_t channel, 
                          uint8_t *found_devices, int max_devices) {
    int device_count = 0;
    
    // First select the channel
    int select_result = pca9546_select_channel(i2c, mux_addr, channel);
    if (select_result != PCA9546_SUCCESS) {
        printf("PCA9546: Failed to select channel 0x%02X for scanning, error %d\n", 
               channel, select_result);
        return select_result;
    }
    
    // Small delay to ensure channel switch completes
    sleep_ms(10);
    
    // Now scan for devices
    printf("PCA9546: Scanning channel 0x%02X for devices...\n", channel);
    
    for (uint8_t addr = 0; addr < 128 && device_count < max_devices; addr++) {
        // Skip reserved I2C addresses and multiplexer address
        if ((addr >= 0x00 && addr <= 0x07) || (addr >= 0x78 && addr <= 0x7F) || (addr == mux_addr)) {
            continue;
        }
        
        uint8_t rxdata;
        int ret = i2c_read_blocking(i2c, addr, &rxdata, 1, false);
        
        if (ret >= 0) {
            printf("PCA9546: Device found at address 0x%02X\n", addr);
            found_devices[device_count++] = addr;
        }
    }
    
    printf("PCA9546: Scan complete, found %d device(s)\n", device_count);
    
    return device_count;
}

const char* pca9546_error_string(int error_code) {
    if (error_code == PCA9546_SUCCESS) {
        return ERROR_STRINGS[0];
    }
    
    // Convert negative error code to index
    int index = -error_code;
    
    if (index >= 1 && index <= 5) {
        return ERROR_STRINGS[index];
    }
    
    return "Unknown error";
}

int pca9546_reset(i2c_inst_t *i2c, uint8_t addr) {
    // Reset just disables all channels and checks communication
    int result = pca9546_disable_all_channels(i2c, addr);
    
    if (result != PCA9546_SUCCESS) {
        return result;
    }
    
    // Verify we can read from the device
    uint8_t value;
    result = pca9546_get_selected_channels(i2c, addr, &value);
    
    if (result != PCA9546_SUCCESS) {
        return result;
    }
    
    if (value != PCA9546_CHANNEL_NONE) {
        return PCA9546_ERROR_VERIFY_FAILED;
    }
    
    return PCA9546_SUCCESS;
}

int pca9546_sleep(i2c_inst_t *i2c, uint8_t addr) {
    // For PCA9546, sleep mode is just disabling all channels
    return pca9546_disable_all_channels(i2c, addr);
}