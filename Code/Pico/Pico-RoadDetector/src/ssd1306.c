#include "hardware_config.h"
#include "ssd1306.h"
#include "ssd1306_font.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// SSD1306 commands
#define SSD1306_SETCONTRAST 0x81
#define SSD1306_DISPLAYALLON_RESUME 0xA4
#define SSD1306_DISPLAYALLON 0xA5
#define SSD1306_NORMALDISPLAY 0xA6
#define SSD1306_INVERTDISPLAY 0xA7
#define SSD1306_DISPLAYOFF 0xAE
#define SSD1306_DISPLAYON 0xAF
#define SSD1306_SETDISPLAYOFFSET 0xD3
#define SSD1306_SETCOMPINS 0xDA
#define SSD1306_SETVCOMDETECT 0xDB
#define SSD1306_SETDISPLAYCLOCKDIV 0xD5
#define SSD1306_SETPRECHARGE 0xD9
#define SSD1306_SETMULTIPLEX 0xA8
#define SSD1306_SETLOWCOLUMN 0x00
#define SSD1306_SETHIGHCOLUMN 0x10
#define SSD1306_SETSTARTLINE 0x40
#define SSD1306_MEMORYMODE 0x20
#define SSD1306_COLUMNADDR 0x21
#define SSD1306_PAGEADDR 0x22
#define SSD1306_COMSCANINC 0xC0
#define SSD1306_COMSCANDEC 0xC8
#define SSD1306_SEGREMAP 0xA0
#define SSD1306_CHARGEPUMP 0x8D
#define SSD1306_EXTERNALVCC 0x01
#define SSD1306_SWITCHCAPVCC 0x02

// Display dimensions
#define SSD1306_PAGES (OLED_HEIGHT / 8)
#define SSD1306_BUFSIZE (OLED_WIDTH * SSD1306_PAGES)

// Static variables
static i2c_inst_t* i2c_instance;
static uint8_t device_addr;
static uint8_t buffer[SSD1306_BUFSIZE];
static const struct render_area full_screen = {
    .start_col = 0,
    .end_col = OLED_WIDTH - 1,
    .start_page = 0,
    .end_page = SSD1306_PAGES - 1
};

// Helper functions
static void send_cmd(uint8_t cmd) {
    uint8_t buf[2] = {0x00, cmd}; // Control byte (0x00) followed by command
    i2c_write_blocking(i2c_instance, device_addr, buf, 2, false);
}

static void send_cmd_list(const uint8_t *cmds, int num_cmds) {
    for (int i = 0; i < num_cmds; i++) {
        send_cmd(cmds[i]);
    }
}

static void render(const struct render_area *area) {
    // Update render area boundaries
    send_cmd(SSD1306_COLUMNADDR);
    send_cmd(area->start_col);
    send_cmd(area->end_col);
    
    send_cmd(SSD1306_PAGEADDR);
    send_cmd(area->start_page);
    send_cmd(area->end_page);
    
    // Calculate buffer size needed
    int buflen = (area->end_col - area->start_col + 1) * (area->end_page - area->start_page + 1);
    
    // Calculate offset into display buffer
    int offset = area->start_page * OLED_WIDTH + area->start_col;
    
    // Send data with control byte (0x40)
    uint8_t *buf = malloc(buflen + 1);
    if (buf) {
        buf[0] = 0x40; // Control byte
        memcpy(buf + 1, &buffer[offset], buflen);
        i2c_write_blocking(i2c_instance, device_addr, buf, buflen + 1, false);
        free(buf);
    }
}

// Draw a single letter
static void draw_letter(uint8_t x, uint8_t y, uint8_t letter_index, uint8_t size) {
    // Each letter in the font array is 8 bytes
    const uint8_t *letter = &font[letter_index * 8];
    
    if (size == 1) {
        // Single size rendering
        for (int col = 0; col < 8; col++) {
            uint8_t line = letter[col];
            for (int row = 0; row < 8; row++) {
                if (line & (1 << row)) {
                    // Convert position to buffer coordinates
                    int buffer_x = x + col;
                    int buffer_y = y + row;
                    
                    if (buffer_x >= 0 && buffer_x < OLED_WIDTH && 
                        buffer_y >= 0 && buffer_y < OLED_HEIGHT) {
                        // Calculate position in buffer
                        int pos = (buffer_y / 8) * OLED_WIDTH + buffer_x;
                        buffer[pos] |= (1 << (buffer_y % 8));
                    }
                }
            }
        }
    } else {
        // Scaled rendering
        for (int col = 0; col < 8; col++) {
            uint8_t line = letter[col];
            for (int row = 0; row < 8; row++) {
                if (line & (1 << row)) {
                    // Draw a size x size block
                    for (int sx = 0; sx < size; sx++) {
                        for (int sy = 0; sy < size; sy++) {
                            int buffer_x = x + col * size + sx;
                            int buffer_y = y + row * size + sy;
                            
                            if (buffer_x >= 0 && buffer_x < OLED_WIDTH && 
                                buffer_y >= 0 && buffer_y < OLED_HEIGHT) {
                                int pos = (buffer_y / 8) * OLED_WIDTH + buffer_x;
                                buffer[pos] |= (1 << (buffer_y % 8));
                            }
                        }
                    }
                }
            }
        }
    }
}

// Initialize the SSD1306 display
void ssd1306_init(i2c_inst_t *i2c_port, uint8_t addr) {
    i2c_instance = i2c_port;
    device_addr = addr;
    
    // Initialize display with common settings for 128x32
    static const uint8_t init_cmds[] = {
        SSD1306_DISPLAYOFF,
        SSD1306_SETDISPLAYCLOCKDIV, 0x80,
        SSD1306_SETMULTIPLEX, 0x1F,  // 0x1F for 128x32, 0x3F for 128x64
        SSD1306_SETDISPLAYOFFSET, 0x00,
        SSD1306_SETSTARTLINE | 0x00,
        SSD1306_CHARGEPUMP, 0x14,    // 0x14 for internal VCC, 0x10 for external
        SSD1306_MEMORYMODE, 0x00,
        SSD1306_SEGREMAP | 0x01,
        SSD1306_COMSCANDEC,
        SSD1306_SETCOMPINS, 0x02,    // 0x02 for 128x32, 0x12 for 128x64
        SSD1306_SETCONTRAST, 0x8F,
        SSD1306_SETPRECHARGE, 0xF1,
        SSD1306_SETVCOMDETECT, 0x40,
        SSD1306_DISPLAYALLON_RESUME,
        SSD1306_NORMALDISPLAY,
        SSD1306_DISPLAYON
    };
    
    send_cmd_list(init_cmds, sizeof(init_cmds));
    
    // Clear the display buffer
    ssd1306_clear();
}

// Clear the display buffer
void ssd1306_clear(void) {
    memset(buffer, 0, SSD1306_BUFSIZE);
}

// Send the buffer to the display
void ssd1306_show(void) {
    render(&full_screen);
}

// Draw a string on the display
void ssd1306_draw_string(uint8_t x, uint8_t y, const char *str, uint8_t size) {
    uint8_t letter_width = 8;  // Each character in the font is 8 pixels wide
    if (size > 1) letter_width *= size;
    
    int i = 0;
    while (str[i] != '\0') {
        if (str[i] >= 'A' && str[i] <= 'Z') {
            draw_letter(x + (i * letter_width), y, str[i] - 'A' + 1, size);
        } 
        else if (str[i] >= 'a' && str[i] <= 'z') {
            // Convert lowercase to uppercase for our font
            draw_letter(x + (i * letter_width), y, (str[i] - 'a') + 1, size);
        }
        else if (str[i] >= '0' && str[i] <= '9') {
            draw_letter(x + (i * letter_width), y, str[i] - '0' + 27, size);
        } 
        else {
            // For spaces and other characters, just advance the cursor
            draw_letter(x + (i * letter_width), y, 0, size);  // Index 0 is blank
        }
        i++;
    }
}