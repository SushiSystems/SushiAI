# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# segment: the mask geometry and the metrics of semantic segmentation.

sushiai_add_module(NAME segment TIER vision
    SOURCES
        source/confusion_matrix.cpp
        source/fill_outline.cpp
        source/intersection_over_union.cpp
        source/pixel_accuracy.cpp
        source/rasterise_outlines.cpp
    PUBLIC_DEPENDS failure vision/imaging
    PRIVATE_DEPENDS seam
    EXTERNAL_PUBLIC SushiRuntime::core)
