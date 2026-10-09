# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# detect: the geometry, assignment, decoding and metrics of detection.

sushiai_add_module(NAME detect TIER vision
    SOURCES
        source/free/anchor_leaves.cpp
        source/free/anchors.cpp
        source/free/assigner.cpp
        source/free/assigner_config.cpp
        source/average_precision.cpp
        source/free/box_coding.cpp
        source/box_format.cpp
        source/detections_nms.cpp
        source/detections_topk.cpp
        source/ground_truth.cpp
        source/iou.cpp
        source/free/level_rows.cpp
        source/free/target_leaves.cpp
        source/top_k.cpp
    PRIVATE_DEPENDS core
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime
    TESTS
        tests/test_detect_anchors.cpp
        tests/test_detect_assigner.cpp
        tests/test_detect_average_precision.cpp
        tests/test_detect_boxes.cpp
        tests/test_detect_postprocess.cpp)
