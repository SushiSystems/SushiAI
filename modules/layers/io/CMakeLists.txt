# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# io: checkpoints and the import of weights into a model state.

sushiai_add_module(NAME io TIER layers
    SOURCES
        source/checkpoint.cpp
        source/checkpoint_format.cpp
        source/import_weights.cpp
    PUBLIC_DEPENDS core nn
    TESTS
        tests/test_checkpoint_format.cpp)
