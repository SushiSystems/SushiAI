# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# generation: the samplers and the loop that continue a prompt token by token.

sushiai_add_module(NAME generation TIER language
    SOURCES
        source/cached_generator.cpp
        source/generation_task.cpp
        source/greedy_sampler.cpp
        source/nucleus_sampler.cpp
        source/position_feed.cpp
        source/ranking.cpp
        source/ring_position_feed.cpp
        source/sampler_section.cpp
        source/table_position_feed.cpp
        source/temperature_sampler.cpp
        source/top_k_sampler.cpp
        source/weights.cpp
        source/window_generator.cpp
    PUBLIC_DEPENDS core config data train language/text
    TESTS
        tests/test_generation_greedy_sampler.cpp
        tests/test_generation_nucleus_sampler.cpp
        tests/test_generation_position_feed.cpp
        tests/test_generation_ring_position_feed.cpp
        tests/test_generation_sampler_section.cpp
        tests/test_generation_table_position_feed.cpp
        tests/test_generation_task.cpp
        tests/test_generation_temperature_sampler.cpp
        tests/test_generation_top_k_sampler.cpp)
