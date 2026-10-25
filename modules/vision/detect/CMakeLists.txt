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
        source/anchor/anchor_task.cpp
        source/anchor/head_training_section.cpp
        source/anchor/assignment_section.cpp
        source/anchor/assignment_thresholds.cpp
        source/anchor/box_delta_iou.cpp
        source/anchor/box_delta_squared.cpp
        source/anchor/head_assignment.cpp
        source/anchor/head_cost.cpp
        source/anchor/head_decode.cpp
        source/anchor/head_delta.cpp
        source/anchor/head_options.cpp
        source/anchor/head_training_options.cpp
        source/anchor/head_truth.cpp
        source/anchor/seeded_sequence.cpp
        source/anchor/shape_ratio_assignment.cpp
        source/anchor/shape_ratio_delta.cpp
        source/average_precision.cpp
        source/batch_average_precision.cpp
        source/box_format.cpp
        source/detections_nms.cpp
        source/detections_topk.cpp
        source/free/anchor_leaves.cpp
        source/free/anchors.cpp
        source/free/assigner.cpp
        source/free/assigner_config.cpp
        source/free/box_coding.cpp
        source/free/distribution_expectation.cpp
        source/free/distribution_targets.cpp
        source/free/free_task.cpp
        source/free/level_rows.cpp
        source/free/target_leaves.cpp
        source/ground_truth.cpp
        source/iou.cpp
        source/iou_gradient.cpp
        source/suppression_criterion.cpp
        source/top_k.cpp
    PUBLIC_DEPENDS config failure
    EXTERNAL_PUBLIC SushiRuntime::core)
