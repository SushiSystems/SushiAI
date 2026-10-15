# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_training_cases: the training step of a Darknet model against its dump.

sushiai_add_module(NAME darknet_training_cases TIER vision INTERNAL
    SOURCES
        source/darknet_training.cpp
    PUBLIC_DEPENDS
        config
        core
        darknet_reference
        engine_fixture
        failure_expectations
        umbrella
        vision/detect
    PRIVATE_DEPENDS
        graph
        nn
        optim
        train
        vision/darknet_forward_cases
        vision/importers
        vision/training)
