# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module eager; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER eager
    SOURCES
        test_partition.cpp
        test_region_boundary.cpp
        test_region_emitter.cpp
        test_region_executor.cpp
        test_region_names.cpp
        test_region_partitioner.cpp
        test_region_program.cpp
        test_region_signature.cpp
        test_region_table.cpp
        test_scope.cpp
        test_scope_partitioner.cpp
        test_scope_trace.cpp
        test_segment.cpp
        test_session.cpp
        test_slot_ledger.cpp
        test_step.cpp
        test_step_executor.cpp
        test_storage_bank.cpp
        test_storage_id.cpp
    DEPENDS
        failure_expectations
        tensor_values)
