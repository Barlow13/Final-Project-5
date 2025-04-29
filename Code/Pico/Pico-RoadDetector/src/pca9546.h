#ifndef PCA9546_H
#define PCA9546_H

#include <stdint.h>
#include <stdbool.h>
#include "hardware/i2c.h"
#include "pico/stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

// Channel bitmasks - each bit corresponds to a channel
#define PCA9546_CHANNEL_0  0x01  // 0001 binary
#define PCA9546_CHANNEL_1  0x02  // 0010 binary
#define PCA9546_CHANNEL_2  0x04  // 0100 binary
#define PCA9546_CHANNEL_3  0x08  // 1000 binary
#define PCA9546_CHANNEL_NONE 0x00 // No channels enabled
#define PCA9546_CHANNEL_ALL  0x0F // All channels enabled

/**
 * @brief Initialize the PCA9546 multiplexer
 * 
 * @param i2c The I2C instance to use (e.g., i2c0, i2c1)
 * @param addr The I2C address of the multiplexer (default: 0x70)
 * @return true if initialization was successful, false otherwise
 */
bool pca9546_init(i2c_inst_t *i2c, uint8_t addr);

/**
 * @brief Simplified function to select a specific port (0-3)
 * 
 * @param i2c The I2C instance to use
 * @param port The port number (0-3)
 * @return true if selection was successful, false otherwise
 */
bool pca9546_select(i2c_inst_t *i2c, uint8_t port);

/**
 * @brief Select a single channel on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask The channel to select (use PCA9546_CHANNEL_x constants)
 * @return true if selection was successful, false otherwise
 */
bool pca9546_select_channel(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);

/**
 * @brief Select multiple channels on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask Bitmask of channels to select (can OR multiple channels together)
 * @return true if selection was successful, false otherwise
 */
bool pca9546_select_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);

/**
 * @brief Read the current channel selection from the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask Pointer to store the channel mask
 * @return true if read was successful, false otherwise
 */
bool pca9546_get_selected_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t *channel_mask);

/**
 * @brief Disable all channels on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @return true if disabling was successful, false otherwise
 */
bool pca9546_disable_all_channels(i2c_inst_t *i2c, uint8_t addr);

/**
 * @brief Scan for I2C devices on a specific multiplexer channel
 * 
 * @param i2c The I2C instance to use
 * @param mux_addr The I2C address of the multiplexer
 * @param channel The channel to scan (use PCA9546_CHANNEL_x constants)
 * @param found_devices Array to store found device addresses
 * @param max_devices Maximum number of devices to find (size of found_devices array)
 * @return Number of devices found
 */
int pca9546_scan_channel(i2c_inst_t *i2c, uint8_t mux_addr, uint8_t channel, 
                       uint8_t *found_devices, int max_devices);

#ifdef __cplusplus
}
#endif

#endif // PCA9546_H