# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# training: the detection objectives and the steps that train them.

sushiai_add_module(NAME training TIER vision
    SOURCES
        source/anchor_bias_init.cpp
        source/anchor_objective.cpp
        source/anchor_objective_factory.cpp
        source/anchor_training_loop.cpp
        source/anchor_training_step.cpp
        source/best_shape_head_delta_rule.cpp
        source/dice_objective.cpp
        source/dice_targets.cpp
        source/distribution_expectation_graph.cpp
        source/distribution_focal_loss.cpp
        source/free_assignment_stage.cpp
        source/free_batch_decoder.cpp
        source/free_bias_init.cpp
        source/free_detection_loss.cpp
        source/free_detection_loss_host.cpp
        source/free_loss_gains.cpp
        source/free_model_facts.cpp
        source/free_objective.cpp
        source/free_objective_factory.cpp
        source/free_training_loop.cpp
        source/free_training_step.cpp
        source/head_delta_loss.cpp
        source/head_delta_rule_choice.cpp
        source/pixel_cross_entropy_objective.cpp
        source/pixel_labels.cpp
        source/pixel_losses.cpp
        source/pixel_objective.cpp
        source/pixel_targets.cpp
        source/segmentation_metrics.cpp
        source/shape_ratio_head_delta_rule.cpp
        source/truth_from_roles.cpp
    PUBLIC_DEPENDS config data dtype failure graph nn optim shape train vision/detect vision/segment
    PRIVATE_DEPENDS execution seam
    EXTERNAL_PUBLIC SushiRuntime::core)
