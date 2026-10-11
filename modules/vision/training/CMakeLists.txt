# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# training: the detection objectives and the steps that train them.

sushiai_add_module(NAME training TIER vision
    SOURCES
        source/anchor_objective.cpp
        source/anchor_objective_factory.cpp
        source/anchor_training_loop.cpp
        source/anchor_training_step.cpp
        source/dice_objective.cpp
        source/dice_targets.cpp
        source/head_delta_loss.cpp
        source/pixel_cross_entropy_objective.cpp
        source/pixel_labels.cpp
        source/pixel_losses.cpp
        source/pixel_objective.cpp
        source/pixel_targets.cpp
        source/segmentation_metrics.cpp
        source/truth_from_roles.cpp
    PUBLIC_DEPENDS core graph config data optim nn train vision/detect vision/segment
    PRIVATE_DEPENDS autograd
    TESTS
        tests/test_anchor_objective.cpp
        tests/test_anchor_objective_factory.cpp
        tests/test_anchor_training_loop.cpp
        tests/test_head_delta_loss.cpp
        tests/test_pixel_objectives.cpp
        tests/test_pixel_targets.cpp
        tests/test_segmentation_metrics.cpp
        tests/test_truth_from_roles.cpp)
