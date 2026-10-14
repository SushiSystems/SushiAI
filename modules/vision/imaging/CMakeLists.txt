# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# imaging: the image type, its transforms and the augmentation pipeline.

sushiai_add_module(NAME imaging TIER vision
    SOURCES
        source/affine.cpp
        source/affine_nearest.cpp
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
        source/mask_pipeline.cpp
        source/mosaic.cpp
        source/pipeline.cpp
        source/polygon.cpp
        source/random.cpp
        source/raw_cache.cpp
        source/recent_partners.cpp
        source/rectangle.cpp
        source/resize.cpp
    PUBLIC_DEPENDS core)
