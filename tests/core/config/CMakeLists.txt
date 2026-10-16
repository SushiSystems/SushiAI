# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module config; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER config
    SOURCES
        test_json_events.cpp
        test_json_writer.cpp
        test_model_config.cpp
        test_model_writer.cpp
        test_registry.cpp
        test_section_reader.cpp
        test_train_config.cpp
        test_value.cpp
    DEPENDS
        failure_expectations
        model_text)
