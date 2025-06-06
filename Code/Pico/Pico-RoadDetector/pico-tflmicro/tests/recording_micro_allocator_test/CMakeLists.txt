cmake_minimum_required(VERSION 3.12)

project(recording_micro_allocator_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(recording_micro_allocator_test "")

target_include_directories(recording_micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test
)

set_target_properties(
  recording_micro_allocator_test
  PROPERTIES
  COMPILE_FLAGS -Os
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

pico_enable_stdio_usb(recording_micro_allocator_test 1)
pico_enable_stdio_uart(recording_micro_allocator_test 0)

target_sources(recording_micro_allocator_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test/recording_micro_allocator_test.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test/tensorflow/lite/micro/testing/test_conv_model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../tests/recording_micro_allocator_test/tensorflow/lite/micro/testing/test_conv_model.h

)

target_link_libraries(
  recording_micro_allocator_test
  pico-tflmicro
  pico-tflmicro_test
)

pico_add_extra_outputs(recording_micro_allocator_test)
