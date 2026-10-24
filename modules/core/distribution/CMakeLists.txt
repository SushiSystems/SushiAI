# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# distribution: the cluster section of a training file, and the split of a batch over replicas.
# Built only when SA_ENABLE_DISTRIBUTED is on; see the root CMakeLists.txt.

sushiai_add_module(NAME distribution TIER core
    SOURCES
        source/cluster_section.cpp
        source/replica_plan.cpp
    PUBLIC_DEPENDS config failure
    PRIVATE_DEPENDS seam
    EXTERNAL_PUBLIC SushiRuntime::placement)
