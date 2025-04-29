# Pico Road Detector

## Wiring

| Device | Interface | GPIO Pins | MUX Channel |
|:-------|:----------|:----------|:------------|
| ArduCAM OV5642 | SPI0 | CS=16 | Channel 2 |
| PCA9546A MUX | I2C1 | SDA=6, SCL=7 | (Always) |
| OLED SSD1306 | I2C1 via MUX | SDA=6, SCL=7 | Channel 1 |

## Build Instructions

```bash
cd Pico-RoadDetector
mkdir build
cd build
cmake ..
make -j4
```

## Flashing
- Boot RP2040 into bootloader (hold BOOTSEL button while plugging USB)
- Drag and drop generated `.uf2` file