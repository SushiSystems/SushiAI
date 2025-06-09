cmake_minimum_required(VERSION 3.10)
project(SushiAI)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

include_directories(core)

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include core)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

add_executable(SushiAI
    main.cpp
    core/tensor.cpp
    core/tensor.h
    core/ops.cpp
    core/ops.cu
    core/ops.h
    core/cuda_utils.h
   )

set_target_properties(SushiAI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)