# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# detect: the geometry, assignment, decoding and metrics of detection.

sushiai_add_module(NAME detect TIER vision
    SOURCES
        source/anchor/anchor_clustering.cpp
        source/anchor/anchor_count.cpp
        source/anchor/anchor_fit.cpp
        source/anchor/anchor_layout.cpp
        source/anchor/anchor_plan.cpp
        source/anchor/anchor_review.cpp
        source/anchor/anchor_size.cpp
        source/anchor/assignment_thresholds.cpp
        source/anchor/box_delta_squared.cpp
        source/anchor/head_assignment.cpp
        source/anchor/head_cost.cpp
        source/anchor/head_decode.cpp
        source/anchor/head_delta.cpp
        source/anchor/head_options.cpp
        source/anchor/head_training_options.cpp
        source/anchor/seeded_sequence.cpp
        source/average_precision.cpp
        source/box_format.cpp
        source/detections_nms.cpp
        source/detections_topk.cpp
        source/free/anchor_leaves.cpp
        source/free/anchors.cpp
        source/free/assigner.cpp
        source/free/assigner_config.cpp
        source/free/box_coding.cpp
        source/free/level_rows.cpp
        source/free/target_leaves.cpp
        source/ground_truth.cpp
        source/iou.cpp
        source/iou_gradient.cpp
        source/suppression_criterion.cpp
        source/top_k.cpp
    PUBLIC_DEPENDS core
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime
    TESTS
        tests/test_detect_anchor_clustering.cpp
        tests/test_detect_anchor_cost.cpp
        tests/test_detect_anchor_count.cpp
        tests/test_detect_anchor_decode.cpp
        tests/test_detect_anchor_delta.cpp
        tests/test_detect_anchor_fit.cpp
        tests/test_detect_anchor_head_assignment.cpp
        tests/test_detect_anchor_layout.cpp
        tests/test_detect_anchor_plan.cpp
        tests/test_detect_anchor_review.cpp
        tests/test_detect_anchor_thresholds.cpp
        tests/test_detect_anchors.cpp
        tests/test_detect_assigner.cpp
        tests/test_detect_average_precision.cpp
        tests/test_detect_boxes.cpp
        tests/test_detect_iou_gradient.cpp
        tests/test_detect_postprocess.cpp
        tests/test_detect_suppression_criterion.cpp)
