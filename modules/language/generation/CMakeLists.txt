# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# generation: the samplers and the loop that continue a prompt token by token.

sushiai_add_module(NAME generation TIER language
    SOURCES
        source/cached_generator.cpp
        source/evaluation_one_position_step.cpp
        source/evaluation_window_logits.cpp
        source/generation_task.cpp
        source/greedy_sampler.cpp
        source/in_memory_generator.cpp
        source/in_memory_one_position_step.cpp
        source/in_memory_window_logits.cpp
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
    PUBLIC_DEPENDS config data dtype failure shape train language/text
    PRIVATE_DEPENDS seam
    EXTERNAL_PUBLIC SushiRuntime::core)
