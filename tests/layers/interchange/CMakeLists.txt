# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module interchange; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER interchange
    SOURCES
        test_axis_translation.cpp
        test_caffe_net.cpp
        test_caffe_writer.cpp
        test_constant_folding.cpp
        test_onnx_graph.cpp
        test_onnx_writer.cpp
        test_tensor_bytes.cpp
    DEPENDS
        failure_expectations)
