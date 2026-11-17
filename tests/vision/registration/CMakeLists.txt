# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module vision/registration; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/registration
    SOURCES
        test_registration/test_registration.cpp
        test_registration/test_ultralytics_files.cpp
        test_registration/test_yolov5n.cpp
        test_registration/test_yolox_x.cpp
        test_registration/test_yolox_x_import.cpp
    DEPENDS
        engine_fixture
        failure_expectations
        layer_rows
        layer_traces
        model_text
        reference_files
        tensor_values)
