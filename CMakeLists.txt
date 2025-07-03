cmake_minimum_required(VERSION 3.10)
project(SushiAI)

set(CMAKE_CXX_STANDARD 17)

include_directories(core)

add_executable(SushiAI
    main.cpp

    core/constants.h

    core/tensor/tensor.h
    core/tensor/tensor.cpp
    core/operations/ops.h
    core/operations/ops.cpp

    core/loss/loss.h
    core/loss/loss.cpp

    core/optimizer/optimizer.h
    core/optimizer/optimizer.cpp

    core/neuralnetwork/initializer.h
    core/neuralnetwork/layer.h
    core/neuralnetwork/layer.cpp
    core/neuralnetwork/sequential.h
    core/neuralnetwork/sequential.cpp

    sushiBLAS/core/backend.h
    sushiBLAS/core/cpu_backend.h
    sushiBLAS/core/cpu_backend.cpp
    sushiBLAS/core/sushiBLAS.cpp
)

add_definitions(-DPROJECT_ROOT_DIR=\"${PROJECT_SOURCE_DIR}\")

target_include_directories(SushiAI PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/loss
    ${PROJECT_SOURCE_DIR}/core/tensor
    ${PROJECT_SOURCE_DIR}/core/operations
    ${PROJECT_SOURCE_DIR}/core/optimizer
    ${PROJECT_SOURCE_DIR}/core/neuralnetwork
    ${PROJECT_SOURCE_DIR}/sushiBLAS
    ${PROJECT_SOURCE_DIR}/sushiBLAS/core
)

