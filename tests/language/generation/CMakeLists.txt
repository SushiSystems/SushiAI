# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module language/generation; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/generation
    SOURCES
        test_cached_generator.cpp
        test_generation_task.cpp
        test_generator.cpp
        test_greedy_sampler.cpp
        test_nucleus_sampler.cpp
        test_position_feed.cpp
        test_ring_position_feed.cpp
        test_sampler.cpp
        test_sampler_section.cpp
        test_table_position_feed.cpp
        test_temperature_sampler.cpp
        test_top_k_sampler.cpp
        test_window_generator.cpp
    DEPENDS
        failure_expectations
        language/decoder_models
        layer_rows
        model_text)
