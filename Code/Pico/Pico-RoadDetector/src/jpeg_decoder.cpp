/**
 * JPEG Decoder Implementation
 * 
 * A lightweight wrapper for the picojpeg library to decode JPEG images
 */
 #include "jpeg_decoder.h"
 #include "picojpeg.h"
 #include <cstdio>
 #include <cstring>
 
 // Buffer for decoded pixels
 static uint8_t g_decoded_buffer[MAX_DECODED_WIDTH * MAX_DECODED_HEIGHT * 3]; // RGB format
 
 // JPEG buffer index for feeding data to decoder
 static uint32_t g_jpeg_buffer_index;
 static uint32_t g_jpeg_buffer_size;
 static const uint8_t* g_jpeg_buffer_data;
 
 // Callback for picojpeg to get more JPEG data
 static unsigned char jpeg_need_bytes_callback(unsigned char* pBuf, unsigned char buf_size, 
                                             unsigned char *pBytes_actually_read, void *pCallback_data) {
     uint32_t bytes_to_read = buf_size;
     
     // Don't read beyond the buffer
     if (g_jpeg_buffer_index + bytes_to_read > g_jpeg_buffer_size)
         bytes_to_read = g_jpeg_buffer_size - g_jpeg_buffer_index;
     
     // Copy data to output buffer
     memcpy(pBuf, &g_jpeg_buffer_data[g_jpeg_buffer_index], bytes_to_read);
     g_jpeg_buffer_index += bytes_to_read;
     
     *pBytes_actually_read = (unsigned char)bytes_to_read;
     
     return 0; // 0 = success
 }
 
 /**
  * Initialize JPEG decoder with buffer data
  */
 bool jpeg_decoder_init(const uint8_t* jpeg_data, uint32_t jpeg_data_size) {
     // Initialize global variables
     g_jpeg_buffer_data = jpeg_data;
     g_jpeg_buffer_size = jpeg_data_size;
     g_jpeg_buffer_index = 0;
     
     // Initialize the decoder
     pjpeg_image_info_t image_info;
     int status = pjpeg_decode_init(&image_info, jpeg_need_bytes_callback, NULL, 0);
     
     if (status != 0) {
         printf("JPEG decoder init failed with status %d\n", status);
         return false;
     }
     
     // Check image dimensions
     if (image_info.m_width > MAX_DECODED_WIDTH || image_info.m_height > MAX_DECODED_HEIGHT) {
         printf("JPEG image too large: %dx%d, max is %dx%d\n", 
                image_info.m_width, image_info.m_height, 
                MAX_DECODED_WIDTH, MAX_DECODED_HEIGHT);
         return false;
     }
     
     printf("JPEG image: %dx%d, components: %d, scan type: %d\n", 
            image_info.m_width, image_info.m_height, 
            image_info.m_comps, image_info.m_scanType);
     
     return true;
 }
 
 /**
  * Decode JPEG image into RGB buffer
  */
 bool jpeg_decode_to_rgb(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                         uint8_t* output_buffer, uint32_t* width, uint32_t* height) {
     // Initialize the decoder
     pjpeg_image_info_t image_info;
     g_jpeg_buffer_data = jpeg_data;
     g_jpeg_buffer_size = jpeg_data_size;
     g_jpeg_buffer_index = 0;
     
     int status = pjpeg_decode_init(&image_info, jpeg_need_bytes_callback, NULL, 0);
     
     if (status != 0) {
         printf("JPEG decoder init failed with status %d\n", status);
         return false;
     }
     
     // Check image dimensions
     if (image_info.m_width > MAX_DECODED_WIDTH || image_info.m_height > MAX_DECODED_HEIGHT) {
         printf("JPEG image too large: %dx%d, max is %dx%d\n", 
                image_info.m_width, image_info.m_height, 
                MAX_DECODED_WIDTH, MAX_DECODED_HEIGHT);
         return false;
     }
     
     // Return dimensions
     *width = image_info.m_width;
     *height = image_info.m_height;
     
     // Decode each MCU block and copy to output buffer
     uint32_t mcu_x = 0;
     uint32_t mcu_y = 0;
     
     for (;;) {
         status = pjpeg_decode_mcu();
         
         if (status == PJPG_NO_MORE_BLOCKS) {
             // Decoding complete
             break;
         }
         else if (status != 0) {
             printf("JPEG decoder error during MCU decode: %d\n", status);
             return false;
         }
         
         // Copy MCU to output buffer
         // The MCU dimensions depend on the image format
         uint32_t mcu_width = image_info.m_MCUWidth;
         uint32_t mcu_height = image_info.m_MCUHeight;
         
         for (uint32_t y = 0; y < mcu_height; y++) {
             // Skip pixels outside image bounds
             if ((mcu_y * mcu_height + y) >= image_info.m_height) continue;
             
             for (uint32_t x = 0; x < mcu_width; x++) {
                 // Skip pixels outside image bounds
                 if ((mcu_x * mcu_width + x) >= image_info.m_width) continue;
                 
                 // Calculate buffer offsets
                 uint32_t src_ofs = y * mcu_width + x;
                 uint32_t dst_ofs = ((mcu_y * mcu_height + y) * image_info.m_width + 
                                    (mcu_x * mcu_width + x)) * 3;
                 
                 // Copy RGB components (JPEG decoder outputs separate color planes)
                 if (image_info.m_scanType == PJPG_GRAYSCALE) {
                     // Grayscale - duplicate Y value to all channels
                     uint8_t gray = image_info.m_pMCUBufR[src_ofs];
                     output_buffer[dst_ofs + 0] = gray; // R
                     output_buffer[dst_ofs + 1] = gray; // G
                     output_buffer[dst_ofs + 2] = gray; // B
                 } 
                 else {
                     // Color image
                     output_buffer[dst_ofs + 0] = image_info.m_pMCUBufR[src_ofs]; // R
                     output_buffer[dst_ofs + 1] = image_info.m_pMCUBufG[src_ofs]; // G
                     output_buffer[dst_ofs + 2] = image_info.m_pMCUBufB[src_ofs]; // B
                 }
             }
         }
         
         // Move to next MCU
         mcu_x++;
         if (mcu_x == image_info.m_MCUSPerRow) {
             mcu_x = 0;
             mcu_y++;
         }
     }
     
     return true;
 }
 
 /**
  * Decode JPEG and downscale to grayscale with specific dimensions
  */
 bool jpeg_decode_to_model_input(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                                 uint8_t* output_buffer, uint32_t target_width, uint32_t target_height) {
     uint32_t width, height;
     
     // Use stack-allocated buffer for RGB data to avoid large static buffer
     uint8_t* rgb_buffer = g_decoded_buffer;
     
     if (!jpeg_decode_to_rgb(jpeg_data, jpeg_data_size, rgb_buffer, &width, &height)) {
         return false;
     }
     
     // Now downscale and convert to grayscale for model input
     for (uint32_t y = 0; y < target_height; y++) {
         for (uint32_t x = 0; x < target_width; x++) {
             // Calculate source coordinate with proper scaling
             uint32_t src_x = x * width / target_width;
             uint32_t src_y = y * height / target_height;
             
             // Ensure we don't go out of bounds
             if (src_x >= width) src_x = width - 1;
             if (src_y >= height) src_y = height - 1;
             
             // Get source pixel
             uint32_t src_ofs = (src_y * width + src_x) * 3;
             
             // Simple RGB to grayscale conversion
             uint8_t r = rgb_buffer[src_ofs + 0];
             uint8_t g = rgb_buffer[src_ofs + 1];
             uint8_t b = rgb_buffer[src_ofs + 2];
             
             // Standard grayscale conversion (luminance)
             uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
             
             // Store to output buffer
             output_buffer[y * target_width + x] = gray;
         }
     }
     
     return true;
 }