# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module decoder_models; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/decoder_models
    SOURCES
        test_decoder_model.cpp
    DEPENDS
        language/decoder_models
        model_text
        failure_expectations)
