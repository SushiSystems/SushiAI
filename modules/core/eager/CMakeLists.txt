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
    PUBLIC_DEPENDS core graph autograd
    TESTS
        tests/test_region_boundary.cpp
        tests/test_region_emitter.cpp
        tests/test_region_executor.cpp
        tests/test_region_program.cpp
        tests/test_region_signature.cpp
        tests/test_region_table.cpp
        tests/test_scope_partitioner.cpp
        tests/test_scope_trace.cpp
        tests/test_session.cpp
        tests/test_slot_ledger.cpp
        tests/test_step_cost.cpp
        tests/test_storage_bank.cpp)
