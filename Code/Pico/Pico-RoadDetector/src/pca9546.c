#include <stdio.h>
#include "hardware_config.h"
#include "pca9546.h"

bool pca9546_init(i2c_inst_t *i2c, uint8_t addr) {
    // Check if the multiplexer is responsive
    uint8_t rxdata;
    int ret = i2c_read_blocking(i2c, addr, &rxdata, 1, false);
    
    // If read was successful, reset all channels to disabled state
    if (ret >= 0) {
        return pca9546_disable_all_channels(i2c, addr);
    }
    
    return false;
}

bool pca9546_select(i2c_inst_t *i2c, uint8_t port) {
    // Wrapper function to simplify channel selection
    // Choose based on port number (0-3)
    if (port > 3) {
        printf("ERROR: Invalid PCA9546 port number %d (must be 0-3)\n", port);
        return false;
    }
    
    uint8_t channel_mask = 1 << port;
    return pca9546_select_channel(i2c, PCA9546_ADDR, channel_mask);
}

bool pca9546_select_channel(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask) {
    // Check that only one channel bit is set (0, 1, 2, or 3)
    if (channel_mask != PCA9546_CHANNEL_0 && 
        channel_mask != PCA9546_CHANNEL_1 && 
        channel_mask != PCA9546_CHANNEL_2 && 
        channel_mask != PCA9546_CHANNEL_3) {
        // For select_channel function, only allow a single channel selection
        printf("ERROR: Invalid channel selection - must select a single channel\n");
        return false;
    }
    
    return pca9546_select_channels(i2c, addr, channel_mask);
}

bool pca9546_select_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask) {
    // Write the channel mask to the multiplexer
    uint8_t buf[1] = { channel_mask };
    int ret = i2c_write_blocking(i2c, addr, buf, 1, false);
    
    if (ret < 0) {
        printf("ERROR: Failed to select MUX channel(s) with code %d\n", ret);
        return false;
    }
    
    // Verify the channel was set correctly (optional)
    uint8_t current_value;
    bool verify_success = pca9546_get_selected_channels(i2c, addr, &current_value);
    
    if (!verify_success) {
        printf("ERROR: Failed to verify MUX channel selection\n");
        return false;
    }
    
    if (current_value != channel_mask) {
        printf("WARNING: MUX channel setting doesn't match requested value!\n");
        printf("  Requested: 0x%02X, Read back: 0x%02X\n", channel_mask, current_value);
        return false;
    }
    
    return true;
}

bool pca9546_get_selected_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t *channel_mask) {
    if (channel_mask == NULL) {
        return false;
    }
    
    int ret = i2c_read_blocking(i2c, addr, channel_mask, 1, false);
    return ret >= 0;
}

bool pca9546_disable_all_channels(i2c_inst_t *i2c, uint8_t addr) {
    return pca9546_select_channels(i2c, addr, PCA9546_CHANNEL_NONE);
}

int pca9546_scan_channel(i2c_inst_t *i2c, uint8_t mux_addr, uint8_t channel, 
                          uint8_t *found_devices, int max_devices) {
    int device_count = 0;
    
    // First select the channel
    if (!pca9546_select_channel(i2c, mux_addr, channel)) {
        printf("Failed to select multiplexer channel 0x%02X\n", channel);
        return 0;
    }
    
    // Small delay to ensure channel switch completes
    sleep_ms(10);
    
    // Now scan for devices
    printf("Scanning multiplexer channel 0x%02X for devices:\n", channel);
    
    for (uint8_t addr = 0; addr < 128 && device_count < max_devices; addr++) {
        // Skip reserved I2C addresses and multiplexer address
        if ((addr >= 0x00 && addr <= 0x07) || (addr >= 0x78 && addr <= 0x7F) || (addr == mux_addr)) {
            continue;
        }
        
        uint8_t rxdata;
        int ret = i2c_read_blocking(i2c, addr, &rxdata, 1, false);
        
        if (ret >= 0) {
            printf("* I2C device found at address 0x%02X\n", addr);
            found_devices[device_count++] = addr;
        }
    }
    
    printf("Scan complete, found %d device(s)\n\n", device_count);
    
    return device_count;
}