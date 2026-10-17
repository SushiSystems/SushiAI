# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_forward_cases: the forward pass of a Darknet model against its dump.

sushiai_add_module(NAME darknet_forward_cases TIER vision INTERNAL
    SOURCES
        source/darknet_forward.cpp
    PUBLIC_DEPENDS
        darknet_reference
        engine_fixture
        failure
        failure_expectations
        umbrella
        vision/detect
        vision/importers
    PRIVATE_DEPENDS allocation config graph nn shape train
    EXTERNAL_PUBLIC SushiRuntime::core)
