# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# training: the next-token objective, the fill of its targets and its metrics.

sushiai_add_module(NAME training TIER language
    SOURCES
        source/expert_balance_objective.cpp
        source/next_token_objective.cpp
        source/next_token_targets.cpp
        source/perplexity.cpp
    PUBLIC_DEPENDS config data failure graph nn train
    PRIVATE_DEPENDS dtype seam shape)
