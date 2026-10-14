# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/blocks; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/blocks
    SOURCES
        test_attention.cpp
        test_bottleneck.cpp
        test_c2psa.cpp
        test_c3k.cpp
        test_c3k2.cpp
        test_conv_norm.cpp
        test_detect_head.cpp
        test_psa_block.cpp
        test_segment_head.cpp
        test_sppf.cpp
    DEPENDS
        failure_expectations)
