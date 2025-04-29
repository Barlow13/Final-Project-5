cmake_minimum_required(VERSION 3.12)

# Pull in PICO SDK (must be before project)
include(pico_sdk_import.cmake)

project(pico-tflmicro C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

pico_sdk_init()

# Create the core TensorFlow Lite Micro library without tests or examples
add_library(pico-tflmicro "")

target_include_directories(pico-tflmicro
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/DSP/Include
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/ruy
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/gemmlowp
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/kissfft
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/flatbuffers
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/Core/Include
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/flatbuffers/include
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Include
)

target_compile_definitions(
  pico-tflmicro
  PUBLIC
  TF_LITE_DISABLE_X86_NEON=1
  TF_LITE_STATIC_MEMORY=1
  CMSIS_NN=1
)

set_target_properties(
  pico-tflmicro
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
)

target_link_libraries(
  pico-tflmicro
  pico_stdlib
)

# Add essential source files only
target_sources(pico-tflmicro
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/c/common.c
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/core/api/error_reporter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/core/api/flatbuffer_conversions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/core/api/op_resolver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/core/api/tensor_utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/kernels/internal/quantization_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/kernels/kernel_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/all_ops_resolver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/activations.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/arg_min_max.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/ceil.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/add.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/conv.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/depthwise_conv.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/fully_connected.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/mul.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/pooling.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/cmsis-nn/softmax.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/comparisons.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/concatenation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/dequantize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/detection_postprocess.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/elementwise.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/ethosu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/floor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/hard_swish.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/kernel_util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/logical.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/logistic.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/maximum_minimum.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/neg.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/pack.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/pad.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/prelu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/quantize.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/reduce.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/reshape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/round.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/shape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/split.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/strided_slice.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/sub.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/svdf.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/tanh.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/kernels/unpack.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/memory_helpers.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/memory_planner/linear_memory_planner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_allocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_error_reporter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_interpreter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_profiler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_string.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/micro_utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/rp2/debug_log.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/rp2/micro_time.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/micro/simple_memory_allocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/tensorflow/lite/schema/schema_utils.cpp
  
  # Essential CMSIS-NN files
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/BasicMathFunctions/arm_elementwise_add_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/NNSupportFunctions/arm_nntables.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/PoolingFunctions/arm_max_pool_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c
  ${CMAKE_CURRENT_LIST_DIR}/src/third_party/cmsis/CMSIS/NN/Source/SoftmaxFunctions/arm_softmax_s8.c
)