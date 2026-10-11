# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# registration: the functions that add the rows of vision to a registry.

sushiai_add_module(NAME registration TIER vision
    SOURCES
        source/registration.cpp
    PUBLIC_DEPENDS core data nn train io
    PRIVATE_DEPENDS vision/blocks vision/datasets vision/detect vision/importers vision/training
    TESTS
        tests/test_datasets_mnist.cpp
        tests/test_registration.cpp)
