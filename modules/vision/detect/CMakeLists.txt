# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# detect: the geometry, assignment, decoding and metrics of detection.

sushiai_add_module(NAME detect TIER vision
    SOURCES
        source/anchor_leaves.cpp
        source/anchors.cpp
        source/assigner.cpp
        source/assigner_config.cpp
        source/average_precision.cpp
        source/box_coding.cpp
        source/box_format.cpp
        source/detections_nms.cpp
        source/detections_topk.cpp
        source/ground_truth.cpp
        source/iou.cpp
        source/level_rows.cpp
        source/target_leaves.cpp
        source/top_k.cpp
    PRIVATE_DEPENDS core
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime
    TESTS
        tests/test_detect_anchors.cpp
        tests/test_detect_assigner.cpp
        tests/test_detect_average_precision.cpp
        tests/test_detect_boxes.cpp
        tests/test_detect_postprocess.cpp)
