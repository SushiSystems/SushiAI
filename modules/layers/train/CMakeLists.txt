# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# train: the objectives, the step builders, the steps and the trainer.

sushiai_add_module(NAME train TIER layers
    SOURCES
        source/batch_role.cpp
        source/binary_confusion.cpp
        source/binary_cross_entropy_objective.cpp
        source/buffer_update_recorder.cpp
        source/decision_threshold.cpp
        source/dense_losses.cpp
        source/dense_objective.cpp
        source/dense_targets.cpp
        source/evaluation_step.cpp
        source/feed.cpp
        source/host_values.cpp
        source/input_match.cpp
        source/mean_squared_error_objective.cpp
        source/metrics.cpp
        source/multi_label_objective.cpp
        source/objective.cpp
        source/objective_context.cpp
        source/objective_registry.cpp
        source/one_hot.cpp
        source/role_match.cpp
        source/step_view.cpp
        source/steps.cpp
        source/sum_objective.cpp
        source/trainer.cpp
        source/training_step.cpp
    PUBLIC_DEPENDS config data dtype failure graph nn optim seam shape
    PRIVATE_DEPENDS autograd execution profile
    EXTERNAL_PUBLIC SushiRuntime::api SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text ${SA_PROFILER_LINKS})
