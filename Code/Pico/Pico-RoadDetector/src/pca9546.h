#ifndef PCA9546_H
#define PCA9546_H

#include <stdint.h>
#include <stdbool.h>
#include "hardware/i2c.h"
#include "pico/stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

// PCA9546 channel definitions
#define PCA9546_CHANNEL_0      0x01  // 0001 binary
#define PCA9546_CHANNEL_1      0x02  // 0010 binary
#define PCA9546_CHANNEL_2      0x04  // 0100 binary
#define PCA9546_CHANNEL_3      0x08  // 1000 binary
#define PCA9546_CHANNEL_NONE   0x00  // No channels enabled
#define PCA9546_CHANNEL_ALL    0x0F  // All channels enabled

// Error codes for more detailed feedback
typedef enum {
    PCA9546_SUCCESS = 0,
    PCA9546_ERROR_COMM_FAILED = -1,   // Communication failure
    PCA9546_ERROR_INVALID_CHANNEL = -2, // Invalid channel specified
    PCA9546_ERROR_VERIFY_FAILED = -3,  // Channel verification failed
    PCA9546_ERROR_DEVICE_BUSY = -4,    // Device is busy
    PCA9546_ERROR_TIMEOUT = -5         // Operation timed out
} pca9546_error_t;

// Configuration options structure
typedef struct {
    uint32_t timeout_ms;       // Timeout in milliseconds for operations
    bool verify_selection;     // Whether to verify channel selection
    uint8_t retry_count;       // Number of retries on failure
} pca9546_config_t;

// Default configuration
extern const pca9546_config_t PCA9546_DEFAULT_CONFIG;

/**
 * @brief Initialize the PCA9546 I2C multiplexer with default configuration
 * 
 * @param i2c The I2C instance to use (e.g., i2c0, i2c1)
 * @param addr The I2C address of the multiplexer (default: 0x70)
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_init(i2c_inst_t *i2c, uint8_t addr);

/**
 * @brief Initialize the PCA9546 with custom configuration
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param config Pointer to configuration options
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_init_with_config(i2c_inst_t *i2c, uint8_t addr, const pca9546_config_t *config);

/**
 * @brief Simplified function to select a specific port (0-3)
 * 
 * @param i2c The I2C instance to use
 * @param port The port number (0-3)
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_select(i2c_inst_t *i2c, uint8_t port);

/**
 * @brief Select a single channel on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask The channel to select (use PCA9546_CHANNEL_x constants)
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_select_channel(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);

/**
 * @brief Select multiple channels on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask Bitmask of channels to select (can OR multiple channels together)
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_select_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);

/**
 * @brief Read the current channel selection from the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @param channel_mask Pointer to store the channel mask
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_get_selected_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t *channel_mask);

/**
 * @brief Disable all channels on the multiplexer
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_disable_all_channels(i2c_inst_t *i2c, uint8_t addr);

/**
 * @brief Scan for I2C devices on a specific multiplexer channel
 * 
 * @param i2c The I2C instance to use
 * @param mux_addr The I2C address of the multiplexer
 * @param channel The channel to scan (use PCA9546_CHANNEL_x constants)
 * @param found_devices Array to store found device addresses
 * @param max_devices Maximum number of devices to find (size of found_devices array)
 * @return Number of devices found, or error code (negative value)
 */
int pca9546_scan_channel(i2c_inst_t *i2c, uint8_t mux_addr, uint8_t channel, 
                          uint8_t *found_devices, int max_devices);

/**
 * @brief Get a string description for an error code
 * 
 * @param error_code The error code to describe
 * @return Pointer to a static string describing the error
 */
const char* pca9546_error_string(int error_code);

/**
 * @brief Reset the PCA9546 device
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_reset(i2c_inst_t *i2c, uint8_t addr);

/**
 * @brief Set the multiplexer to low power mode
 * (All channels disabled, minimal current consumption)
 * 
 * @param i2c The I2C instance to use
 * @param addr The I2C address of the multiplexer
 * @return PCA9546_SUCCESS if successful, error code otherwise
 */
int pca9546_sleep(i2c_inst_t *i2c, uint8_t addr);

#ifdef __cplusplus
}
#endif

#endif // PCA9546_H