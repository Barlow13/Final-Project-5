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
int pca9546_init(i2c_inst_t *i2c, uint8_t addr);
int pca9546_init_with_config(i2c_inst_t *i2c, uint8_t addr, const pca9546_config_t *config);
int pca9546_select(i2c_inst_t *i2c, uint8_t port);
int pca9546_select_channel(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);
int pca9546_select_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t channel_mask);
int pca9546_get_selected_channels(i2c_inst_t *i2c, uint8_t addr, uint8_t *channel_mask);
int pca9546_disable_all_channels(i2c_inst_t *i2c, uint8_t addr);
int pca9546_scan_channel(i2c_inst_t *i2c, uint8_t mux_addr, uint8_t channel, 
                          uint8_t *found_devices, int max_devices);
const char* pca9546_error_string(int error_code);
int pca9546_reset(i2c_inst_t *i2c, uint8_t addr);
int pca9546_sleep(i2c_inst_t *i2c, uint8_t addr);

#ifdef __cplusplus
}
#endif

#endif // PCA9546_H