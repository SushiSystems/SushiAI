# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_training_cases: the training step of a Darknet model against its dump.

sushiai_add_internal_module(NAME darknet_training_cases TIER vision
    SOURCES
        source/darknet_training.cpp
    DEPENDS
        umbrella
        vision/detect
        vision/importers
        vision/training
        vision/darknet_forward_cases
        darknet_reference
        engine_fixture
        failure_expectations
        tensor_values)
