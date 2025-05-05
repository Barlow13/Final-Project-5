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


void ssd1306_init(i2c_inst_t *i2c_port, uint8_t addr);
void ssd1306_clear(void);
void ssd1306_show(void);
void ssd1306_draw_string(uint8_t x, uint8_t y, const char *str, uint8_t size);
void ssd1306_draw_pixel(uint8_t x, uint8_t y, uint8_t color);
void ssd1306_draw_hline(uint8_t x, uint8_t y, uint8_t width, uint8_t color);
void ssd1306_draw_rect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);
void ssd1306_fill_rect(uint8_t x, uint8_t y, uint8_t width, uint8_t height, uint8_t color);

#ifdef __cplusplus
}
#endif

#endif // SSD1306_H