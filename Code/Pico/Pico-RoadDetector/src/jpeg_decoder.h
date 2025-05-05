/**
 * JPEG Decoder Header
 * 
 * Interface for decoding JPEG images using the picojpeg library
 */

 #ifndef JPEG_DECODER_H
 #define JPEG_DECODER_H
 
 #include <stdint.h>
 #include <stdbool.h>
 
 // Maximum decoded image dimensions
 #define MAX_DECODED_WIDTH  64
 #define MAX_DECODED_HEIGHT 64
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 /**
  * Initialize JPEG decoder with buffer data
  */
 bool jpeg_decoder_init(const uint8_t* jpeg_data, uint32_t jpeg_data_size);
 
 /**
  * Decode JPEG image into RGB buffer
  */
 bool jpeg_decode_to_rgb(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                         uint8_t* output_buffer, uint32_t* width, uint32_t* height);
 
 /**
  * Decode JPEG and downscale to grayscale with specific dimensions
  * Useful for preparing input to machine learning models
  */
 bool jpeg_decode_to_model_input(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                                uint8_t* output_buffer, uint32_t target_width, uint32_t target_height);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // JPEG_DECODER_H