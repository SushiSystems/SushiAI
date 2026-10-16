# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module autograd; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER autograd
    SOURCES
        rules/test_activation.cpp
        rules/test_all.cpp
        rules/test_elementwise.cpp
        rules/test_linalg.cpp
        rules/test_loss.cpp
        rules/test_norm.cpp
        rules/test_random.cpp
        rules/test_shape.cpp
        rules/test_spatial.cpp
        test_backward.cpp
        test_gradcheck.cpp
        test_provenance.cpp
        test_rule.cpp
        test_tape.cpp
    DEPENDS
        failure_expectations
        seeded_noise
        tensor_values)
