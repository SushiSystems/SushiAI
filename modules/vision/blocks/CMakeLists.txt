# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# blocks: the model blocks of the YOLO family, built from nn layers.

sushiai_add_module(NAME blocks TIER vision
    SOURCES
        source/bottleneck.cpp
        source/c3k.cpp
        source/c3k2.cpp
        source/conv_norm.cpp
        source/sppf.cpp
    PUBLIC_DEPENDS core graph nn
    PRIVATE_DEPENDS config
    TESTS
        tests/test_block_bottleneck.cpp
        tests/test_block_c3k.cpp
        tests/test_block_c3k2.cpp
        tests/test_block_conv_norm.cpp
        tests/test_block_sppf.cpp)
