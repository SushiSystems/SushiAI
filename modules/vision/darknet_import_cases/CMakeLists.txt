# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_import_cases: the import of a Darknet model against its dump.

sushiai_add_internal_module(NAME darknet_import_cases TIER vision
    SOURCES
        source/darknet_import.cpp
    DEPENDS
        umbrella
        vision/detect
        vision/importers
        vision/darknet_forward_cases
        darknet_reference
        engine_fixture
        failure_expectations
        tensor_values)
