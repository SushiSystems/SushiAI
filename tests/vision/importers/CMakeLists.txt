# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/importers; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/importers
    SOURCES
        test_darknet_config.cpp
        test_darknet_fold.cpp
        test_darknet_import/test_decode.cpp
        test_darknet_import/test_import.cpp
        test_darknet_import/test_yolov3_forward.cpp
        test_darknet_import/test_yolov3_tiny_forward.cpp
        test_darknet_import/test_yolov3_tiny_layers.cpp
        test_darknet_import/test_yolov4_forward.cpp
        test_darknet_import/test_yolov4_tiny_forward.cpp
        test_darknet_import/test_yolov7_forward.cpp
        test_darknet_import/test_yolov7_tiny_forward.cpp
        test_darknet_model.cpp
        test_darknet_recipe.cpp
        test_darknet_store.cpp
        test_darknet_tensors.cpp
        test_darknet_training/test_recipe.cpp
        test_darknet_training/test_yolov3.cpp
        test_darknet_training/test_yolov3_tiny.cpp
        test_darknet_training/test_yolov4.cpp
        test_darknet_training/test_yolov4_tiny.cpp
        test_darknet_training/test_yolov7.cpp
        test_darknet_training/test_yolov7_tiny.cpp
        test_darknet_weights.cpp
    DEPENDS
        darknet_reference
        failure_expectations
        layer_traces
        scratch_files
        vision/darknet_forward_cases
        vision/darknet_import_cases
        vision/darknet_training_cases)
