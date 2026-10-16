# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module language/blocks; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/blocks
    SOURCES
        test_causal_self_attention.cpp
        test_expert_projection.cpp
        test_expert_trace.cpp
        test_expert_transformer_block.cpp
        test_feed_forward.cpp
        test_incremental.cpp
        test_language_head.cpp
        test_moe_feed_forward.cpp
        test_position_embedding.cpp
        test_rotary_embedding.cpp
        test_rotary_positions.cpp
        test_transformer_block.cpp
    DEPENDS
        failure_expectations
        seeded_noise
        tensor_values)
