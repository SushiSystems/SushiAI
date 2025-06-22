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
    optim/optimizer.cpp
    optim/optimizer.h
    nn/sequential.cpp
    nn/initializer.h
    nn/sequential.h
    core/constants.h
    core/tensor.cpp
    core/tensor.h
    loss/loss.cpp
    loss/loss.h
    core/ops.cpp
    core/ops.h
    nn/layer.cpp
    nn/layer.h)

target_include_directories(SushiAI PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/nn
    ${PROJECT_SOURCE_DIR}/optim
    ${PROJECT_SOURCE_DIR}/loss
)

target_link_libraries(SushiAI cudart)
set_target_properties(SushiAI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
