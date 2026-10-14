# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module graph; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER graph
    SOURCES
        lowering/test_all.cpp
        lowering/test_random.cpp
        test_builder.cpp
        test_compiled_step.cpp
        test_dropout_mask.cpp
        test_fusion.cpp
        test_ir.cpp
        test_memory_plan.cpp
        test_step_recorder.cpp
        test_traced.cpp
    DEPENDS
        failure_expectations
        tensor_values)
