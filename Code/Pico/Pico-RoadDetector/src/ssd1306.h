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

/**
 * @brief Draw a single pixel on the display
 * 
 * @param x X position (0-127)
 * @param y Y position (0-31)
 * @param color 1 for pixel on, 0 for pixel off
 */
void ssd1306_draw_pixel(uint8_t x, uint8_t y, uint8_t color);

/**
 * @brief Draw a horizontal line
 * 
 * @param x Start X position
 * @param y Y position
 * @param width Line width in pixels
 * @param color 1 for pixel on, 0 for pixel off
 */
void ssd1306_draw_hline(uint8_t x, uint8_t y, uint8_t width, uint8_t color);

/**
 * @brief Draw a rectangle outline
 * 
 * @param x X position of top-left corner
 * @param y Y position of top-left corner
 * @param width Width in pixels
 * @param height Height in pixels
 * @param color 1 for pixel on, 0 for pixel off
 */
void ssd1306_draw_rect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);

/**
 * @brief Draw a filled rectangle
 * 
 * @param x X position of top-left corner
 * @param y Y position of top-left corner
 * @param width Width in pixels
 * @param height Height in pixels
 * @param color 1 for pixel on, 0 for pixel off
 */
void ssd1306_fill_rect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);

#ifdef __cplusplus
}
#endif

#endif // SSD1306_H