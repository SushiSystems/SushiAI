# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/training; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/training
    SOURCES
        test_anchor_bias_init.cpp
        test_anchor_objective.cpp
        test_anchor_objective_factory.cpp
        test_anchor_training_loop.cpp
        test_anchor_training_step.cpp
        test_dice_objective.cpp
        test_dice_targets.cpp
        test_distribution_expectation_graph.cpp
        test_distribution_focal_loss.cpp
        test_free_assignment_stage.cpp
        test_free_batch_decoder.cpp
        test_free_bias_init.cpp
        test_free_detection_loss.cpp
        test_free_loss_gains.cpp
        test_free_model_facts.cpp
        test_free_objective.cpp
        test_free_training_step.cpp
        test_head_delta_loss.cpp
        test_pixel_cross_entropy_objective.cpp
        test_pixel_extents.cpp
        test_pixel_labels.cpp
        test_pixel_losses.cpp
        test_pixel_targets.cpp
        test_segmentation_metrics.cpp
        test_truth_from_roles.cpp
    DEPENDS
        engine_fixture
        failure_expectations
        seeded_noise
        tensor_values
        vision/blocks)
