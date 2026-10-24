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
        source/replica_accumulator.cpp
        source/replica_draws.cpp
        source/replica_kernel.cpp
        source/replica_plan.cpp
        source/replica_step.cpp
        source/replica_task.cpp
    PUBLIC_DEPENDS config failure graph
    PRIVATE_DEPENDS seam execution
    EXTERNAL_PUBLIC SushiBLAS::sushiblas SushiRuntime::placement SushiRuntime::distributed)
