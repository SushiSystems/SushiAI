# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/training; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/training
    SOURCES
        test_anchor_objective.cpp
        test_anchor_objective_factory.cpp
        test_anchor_training_loop.cpp
        test_anchor_training_step.cpp
        test_dice_objective.cpp
        test_head_delta_loss.cpp
        test_pixel_cross_entropy_objective.cpp
        test_pixel_extents.cpp
        test_pixel_targets.cpp
        test_segmentation_metrics.cpp
        test_truth_from_roles.cpp
    DEPENDS
        failure_expectations
        seeded_noise
        tensor_values)
