# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module vision/exporters; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/exporters
    SOURCES
        test_caffe_export.cpp
        test_darknet_export.cpp
        test_onnx_export.cpp
    DEPENDS
        engine_fixture
        failure_expectations
        scratch_files
        vision/darknet_import_cases
        vision/importers)
