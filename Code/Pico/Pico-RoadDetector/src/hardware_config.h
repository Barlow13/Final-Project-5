/**
 * @file hardware_config.h
 * @brief Central hardware configuration for Pico-RoadDetector
 * @author Created by Claude
 * @date April 28, 2025
 */

 #ifndef HARDWARE_CONFIG_H
 #define HARDWARE_CONFIG_H
 
 #include "pico/stdlib.h"
 #include "hardware/i2c.h"
 #include "hardware/spi.h"
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /* SPI Configuration for ArduCAM */
 #define SPI_PORT        spi0
 #define PIN_SCK         2
 #define PIN_MOSI        3
 #define PIN_MISO        4
 #define PIN_CS          16
 #define ARDUCAM_SPI_FREQ  (1 * 1000 * 1000)  // 4 MHz
 
 /* I2C Configuration */
 #define I2C_PORT        i2c1
 #define PIN_SDA         6
 #define PIN_SCL         7
 #define I2C_FREQ        (100 * 1000)  // 100 kHz for camera
 
 /* I2C Device Addresses */
 #define SSD1306_ADDR    0x3C
 #define PCA9546_ADDR    0x70
 
 /* PCA9546 Multiplexer Ports */
 #define MUX_PORT_ARDUCAM    2
 #define MUX_PORT_OLED       1

 
 /* OLED Display Configuration */
 #define OLED_WIDTH      128
 #define OLED_HEIGHT     32
 
 /* UART Configuration for debug output */
 #define UART_ID         uart0
 #define BAUD_RATE       921600
 #define DATA_BITS       8
 #define STOP_BITS       1
 #define PARITY          UART_PARITY_NONE
 #define UART_TX_PIN     0
 #define UART_RX_PIN     1
 
 /* ArduChip registers definition */
 #define ARDUCHIP_TEST1        0x00  // TEST register
 #define ARDUCHIP_FIFO         0x04  // FIFO and I2C control
 #define FIFO_CLEAR_MASK       0x01
 #define FIFO_START_MASK       0x02
 #define FIFO_RDPTR_RST_MASK   0x10
 #define FIFO_WRPTR_RST_MASK   0x20
 
 #define ARDUCHIP_TRIG         0x41  // Trigger source
 #define CAP_DONE_MASK         0x08
 
 #define BURST_FIFO_READ       0x3C  // Burst FIFO read operation
 #define SINGLE_FIFO_READ      0x3D  // Single FIFO read operation
 
 #define FIFO_SIZE1            0x42  // Camera write FIFO size[7:0] for burst to read
 #define FIFO_SIZE2            0x43  // Camera write FIFO size[15:8]
 #define FIFO_SIZE3            0x44  // Camera write FIFO size[18:16]
 
 #define WRITE_BIT             0x80
 
 /* ArduCAM helper macros */
 #define cbi(reg, bitmask) gpio_put(bitmask, 0)
 #define sbi(reg, bitmask) gpio_put(bitmask, 1)
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* HARDWARE_CONFIG_H */