# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# imaging: the image type, its transforms and the augmentation pipeline.

sushiai_add_module(NAME imaging TIER vision
    SOURCES
        source/affine.cpp
        source/bilinear.cpp
        source/boxes.cpp
        source/dataset_partners.cpp
        source/flip.cpp
        source/gray.cpp
        source/hsv.cpp
        source/image.cpp
        source/label_set.cpp
        source/labels.cpp
        source/letterbox.cpp
        source/mosaic.cpp
        source/pipeline.cpp
        source/polygon.cpp
        source/random.cpp
        source/raw_cache.cpp
        source/recent_partners.cpp
        source/rectangle.cpp
        source/resize.cpp
    PRIVATE_DEPENDS core
    TESTS
        tests/test_imaging_affine.cpp
        tests/test_imaging_boxes.cpp
        tests/test_imaging_flip.cpp
        tests/test_imaging_gray.cpp
        tests/test_imaging_hsv.cpp
        tests/test_imaging_image.cpp
        tests/test_imaging_label_set.cpp
        tests/test_imaging_labels.cpp
        tests/test_imaging_letterbox.cpp
        tests/test_imaging_mosaic.cpp
        tests/test_imaging_partners.cpp
        tests/test_imaging_pipeline.cpp
        tests/test_imaging_polygon.cpp
        tests/test_imaging_random.cpp
        tests/test_imaging_raw_cache.cpp
        tests/test_imaging_rectangle.cpp
        tests/test_imaging_resize.cpp)
