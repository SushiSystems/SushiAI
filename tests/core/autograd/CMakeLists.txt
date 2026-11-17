# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module autograd; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER autograd
    SOURCES
        compile_rule.cpp
        compile_tape.cpp
        rules/compile_elementwise.cpp
        rules/test_activation/test_gradients.cpp
        rules/test_activation/test_rules.cpp
        rules/test_all.cpp
        rules/test_broadcast.cpp
        rules/test_elementwise/test_gradients.cpp
        rules/test_elementwise/test_rules.cpp
        rules/test_linalg/test_gradients.cpp
        rules/test_linalg/test_rules.cpp
        rules/test_loss/test_gradients.cpp
        rules/test_loss/test_rules.cpp
        rules/test_norm/test_gradients.cpp
        rules/test_norm/test_rules.cpp
        rules/test_random.cpp
        rules/test_shape/test_gradients.cpp
        rules/test_shape/test_rules.cpp
        rules/test_spatial/test_gradients.cpp
        rules/test_spatial/test_rules.cpp
        test_backward.cpp
        test_gradcheck.cpp
        test_provenance.cpp
        test_rule.cpp
        test_tape.cpp
    DEPENDS
        failure_expectations
        seeded_noise
        tensor_values)
