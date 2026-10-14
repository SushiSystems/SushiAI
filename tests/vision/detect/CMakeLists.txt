# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/detect; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/detect
    SOURCES
        anchor/test_anchor_clustering.cpp
        anchor/test_anchor_count.cpp
        anchor/test_anchor_fit.cpp
        anchor/test_anchor_layout.cpp
        anchor/test_anchor_plan.cpp
        anchor/test_anchor_review.cpp
        anchor/test_anchor_task.cpp
        anchor/test_assignment_thresholds.cpp
        anchor/test_head_assignment.cpp
        anchor/test_head_cost.cpp
        anchor/test_head_decode.cpp
        anchor/test_head_delta.cpp
        anchor/test_head_options.cpp
        anchor/test_head_training_options.cpp
        anchor/test_head_training_section.cpp
        anchor/test_head_truth.cpp
        free/test_anchors.cpp
        free/test_assigner.cpp
        free/test_box_coding.cpp
        test_average_precision.cpp
        test_box_format.cpp
        test_detections_nms.cpp
        test_detections_topk.cpp
        test_iou.cpp
        test_iou_gradient.cpp
        test_suppression_criterion.cpp
        test_top_k.cpp
    DEPENDS
        failure_expectations
        reference_files)
