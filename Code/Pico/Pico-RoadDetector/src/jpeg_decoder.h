/**
 * JPEG Decoder Header
 * 
 * Interface for decoding JPEG images using the picojpeg library
 * for the Raspberry Pi Pico road detection project.
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
  * 
  * @param jpeg_data Pointer to JPEG data buffer
  * @param jpeg_data_size Size of JPEG data in bytes
  * @return true if initialization succeeded, false otherwise
  */
 bool jpeg_decoder_init(const uint8_t* jpeg_data, uint32_t jpeg_data_size);
 
 /**
  * Decode JPEG image into RGB buffer
  * 
  * @param jpeg_data Pointer to JPEG data buffer
  * @param jpeg_data_size Size of JPEG data in bytes
  * @param output_buffer Buffer for RGB output (must be width*height*3 bytes)
  * @param width Pointer to store decoded image width
  * @param height Pointer to store decoded image height
  * @return true if decoding succeeded, false otherwise
  */
 bool jpeg_decode_to_rgb(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                         uint8_t* output_buffer, uint32_t* width, uint32_t* height);
 
 /**
  * Decode JPEG and downscale to grayscale with specific dimensions
  * Useful for preparing input to machine learning models
  * 
  * @param jpeg_data Pointer to JPEG data buffer
  * @param jpeg_data_size Size of JPEG data in bytes
  * @param output_buffer Buffer for grayscale output (must be target_width*target_height bytes)
  * @param target_width Desired output width
  * @param target_height Desired output height
  * @return true if decoding succeeded, false otherwise
  */
 bool jpeg_decode_to_model_input(const uint8_t* jpeg_data, uint32_t jpeg_data_size, 
                                uint8_t* output_buffer, uint32_t target_width, uint32_t target_height);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif // JPEG_DECODER_H