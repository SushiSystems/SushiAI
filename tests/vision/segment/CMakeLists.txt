# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/segment; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/segment
    SOURCES
        test_argmax_classes.cpp
        test_confusion_matrix.cpp
        test_fill_outline.cpp
        test_intersection_over_union.cpp
        test_pixel_accuracy.cpp
        test_rasterise_outlines.cpp
    DEPENDS
        failure_expectations)
