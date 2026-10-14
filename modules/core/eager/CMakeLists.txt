# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# eager: the host-lane units of the eager front end.

sushiai_add_module(NAME eager TIER core
    SOURCES
        source/region_boundary.cpp
        source/region_emitter.cpp
        source/region_executor.cpp
        source/region_names.cpp
        source/region_program.cpp
        source/region_signature.cpp
        source/region_table.cpp
        source/scope_partitioner.cpp
        source/scope_trace.cpp
        source/session.cpp
        source/slot_ledger.cpp
        source/step.cpp
        source/storage_bank.cpp
    PUBLIC_DEPENDS core graph autograd)
