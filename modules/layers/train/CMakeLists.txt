# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# train: the objectives, the step builders, the steps and the trainer.

sushiai_add_module(NAME train TIER layers
    SOURCES
        source/buffer_update_recorder.cpp
        source/evaluation_step.cpp
        source/feed.cpp
        source/objective.cpp
        source/steps.cpp
        source/trainer.cpp
        source/training_step.cpp
    PUBLIC_DEPENDS core graph config data optim nn
    PRIVATE_DEPENDS autograd
    TESTS
        tests/test_objective.cpp
        tests/test_steps.cpp)
