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
        anchor/test_anchor_size.cpp
        anchor/test_anchor_task.cpp
        anchor/test_assignment_section.cpp
        anchor/test_assignment_thresholds.cpp
        anchor/test_head_assignment.cpp
        anchor/test_head_cost.cpp
        anchor/test_head_decode.cpp
        anchor/test_head_delta.cpp
        anchor/test_head_options.cpp
        anchor/test_head_training_options.cpp
        anchor/test_head_training_section.cpp
        anchor/test_head_truth.cpp
        anchor/test_seeded_sequence.cpp
        anchor/test_shape_ratio_assignment.cpp
        anchor/test_shape_ratio_delta.cpp
        anchor/test_target_assignment.cpp
        free/test_anchor_leaves.cpp
        free/test_anchors.cpp
        free/test_assigner.cpp
        free/test_assigner_config.cpp
        free/test_assignment.cpp
        free/test_box_coding.cpp
        free/test_distribution_expectation.cpp
        free/test_distribution_targets.cpp
        free/test_free_task.cpp
        free/test_head_contract.cpp
        free/test_level_rows.cpp
        free/test_target_leaves.cpp
        test_average_precision.cpp
        test_batch_average_precision.cpp
        test_box_format.cpp
        test_detection_row.cpp
        test_detections_nms.cpp
        test_detections_topk.cpp
        test_float_order.cpp
        test_ground_truth.cpp
        test_iou.cpp
        test_iou_gradient.cpp
        test_prediction_shape.cpp
        test_suppression_criterion.cpp
        test_top_k.cpp
    DEPENDS
        failure_expectations
        reference_files)
