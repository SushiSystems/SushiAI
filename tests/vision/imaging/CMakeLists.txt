# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/imaging; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/imaging
    SOURCES
        test_affine.cpp
        test_boxes.cpp
        test_dataset_partners.cpp
        test_flip.cpp
        test_gray.cpp
        test_hsv.cpp
        test_image.cpp
        test_label_set.cpp
        test_labels.cpp
        test_letterbox.cpp
        test_mask_pipeline.cpp
        test_mosaic.cpp
        test_pipeline.cpp
        test_polygon.cpp
        test_random.cpp
        test_raw_cache.cpp
        test_recent_partners.cpp
        test_rectangle.cpp
        test_resize.cpp
    DEPENDS
        failure_expectations)
