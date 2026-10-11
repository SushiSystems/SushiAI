# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# blocks: the model blocks of a transformer over a token sequence, built from nn layers.

sushiai_add_module(NAME blocks TIER language
    SOURCES
        source/causal_self_attention.cpp
        source/expert_projection.cpp
        source/expert_transformer_block.cpp
        source/feed_forward.cpp
        source/language_head.cpp
        source/moe_feed_forward.cpp
        source/position_embedding.cpp
        source/rotary_embedding.cpp
        source/rotary_positions.cpp
        source/transformer_block.cpp
    PUBLIC_DEPENDS core graph nn
    PRIVATE_DEPENDS config
    TESTS
        tests/test_block_causal_self_attention.cpp
        tests/test_block_expert_transformer_block.cpp
        tests/test_block_feed_forward.cpp
        tests/test_block_language_head.cpp
        tests/test_block_moe_feed_forward.cpp
        tests/test_block_position_embedding.cpp
        tests/test_block_rotary_embedding.cpp
        tests/test_block_rotary_positions.cpp
        tests/test_block_transformer_block.cpp)
