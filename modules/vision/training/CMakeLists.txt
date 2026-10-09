# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# training: the detection objectives and the steps that train them.

sushiai_add_module(NAME training TIER vision
    SOURCES
        source/anchor_objective.cpp
        source/anchor_training_step.cpp
        source/head_delta_loss.cpp
    PUBLIC_DEPENDS core graph config data optim nn train vision/detect
    PRIVATE_DEPENDS autograd
    TESTS
        tests/test_anchor_objective.cpp
        tests/test_head_delta_loss.cpp)
