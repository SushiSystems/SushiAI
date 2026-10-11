# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# segment: the mask geometry and the metrics of semantic segmentation.

sushiai_add_module(NAME segment TIER vision
    SOURCES
        source/confusion_matrix.cpp
        source/fill_outline.cpp
        source/intersection_over_union.cpp
        source/pixel_accuracy.cpp
        source/rasterise_outlines.cpp
    PUBLIC_DEPENDS core vision/imaging
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime
    TESTS
        tests/test_segment_confusion_matrix.cpp
        tests/test_segment_metrics.cpp
        tests/test_segment_rasterise.cpp)
