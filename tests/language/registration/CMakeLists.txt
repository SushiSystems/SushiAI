# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module language/registration; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/registration
    SOURCES
        test_registration.cpp
    DEPENDS
        failure_expectations
        language/decoder_models
        layer_rows
        model_text
        seeded_noise
        tensor_values)
