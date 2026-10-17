# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_import_cases: the import of a Darknet model against its dump.

sushiai_add_module(NAME darknet_import_cases TIER vision INTERNAL
    SOURCES
        source/darknet_import.cpp
    PUBLIC_DEPENDS
        engine_fixture
        failure
        failure_expectations
        io
        nn
        umbrella
        vision/importers
    PRIVATE_DEPENDS
        config
        darknet_reference
        dtype
        shape
        train
        vision/darknet_forward_cases
        vision/detect
    EXTERNAL_PUBLIC SushiRuntime::core)
