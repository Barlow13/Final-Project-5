#ifndef SSD1306_H
#define SSD1306_H

#include <stdbool.h>
#include <stdint.h>
#include "hardware/i2c.h"
#include "hardware_config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct render_area {
    uint8_t start_col;
    uint8_t end_col;
    uint8_t start_page;
    uint8_t end_page;
};

/**
 * @brief Initialize the SSD1306 OLED display
 * 
 * @param i2c_port The I2C port to use (e.g., i2c0, i2c1)
 * @param addr The I2C address of the display (typically 0x3C or 0x3D)
 */
void ssd1306_init(i2c_inst_t *i2c_port, uint8_t addr);

/**
 * @brief Clear the display buffer (doesn't update the display)
 */
void ssd1306_clear(void);

/**
 * @brief Send the buffer contents to the display
 */
void ssd1306_show(void);

/**
 * @brief Draw a string on the display
 * 
 * @param x X position (0-127)
 * @param y Y position (0-31 for 128x32 display)
 * @param str String to display (uppercase A-Z and numbers 0-9 supported)
 * @param size Size multiplier (1 = normal, 2 = double size, etc.)
 */
void ssd1306_draw_string(uint8_t x, uint8_t y, const char *str, uint8_t size);

#ifdef __cplusplus
}
#endif

#endif // SSD1306_H