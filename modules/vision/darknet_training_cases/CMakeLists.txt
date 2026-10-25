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
        darknet_reference
        engine_fixture
        failure
        failure_expectations
        vision/detect
    PRIVATE_DEPENDS
        allocation
        graph
        nn
        ops
        optim
        shape
        train
        vision/darknet_forward_cases
        vision/importers
        vision/training
    EXTERNAL_PUBLIC SushiRuntime::core)
