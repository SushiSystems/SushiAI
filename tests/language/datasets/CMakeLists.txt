# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module language/datasets; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER language/datasets
    SOURCES
        test_next_token_collator.cpp
        test_prompt_dataset.cpp
        test_token_dataset.cpp
        test_token_file.cpp
        test_tokens_section.cpp
        test_window_layout.cpp
    DEPENDS
        failure_expectations
        layer_rows)
