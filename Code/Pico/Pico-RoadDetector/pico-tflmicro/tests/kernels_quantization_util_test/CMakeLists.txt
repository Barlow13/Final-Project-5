cmake_minimum_required(VERSION 3.12)

project(kernels_quantization_util_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(kernels_quantization_util_test "")

target_include_directories(kernels_quantization_util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_quantization_util_test
)

set_target_properties(
  kernels_quantization_util_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(kernels_quantization_util_test 1)
pico_enable_stdio_uart(kernels_quantization_util_test 0)

target_sources(kernels_quantization_util_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/kernels_quantization_util_test/quantization_util_test.cpp

)

target_link_libraries(
  kernels_quantization_util_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(kernels_quantization_util_test)
