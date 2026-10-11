# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# datasets: the token file of a corpus and the dataset that reads it in windows.

sushiai_add_module(NAME datasets TIER language
    SOURCES
        source/disk.cpp
        source/next_token_collator.cpp
        source/prompt_dataset.cpp
        source/token_dataset.cpp
        source/token_file.cpp
        source/tokens_section.cpp
        source/window_layout.cpp
    PUBLIC_DEPENDS core config data language/text
    TESTS
        tests/test_datasets_next_token_collator.cpp
        tests/test_datasets_prompt_dataset.cpp
        tests/test_datasets_token_dataset.cpp
        tests/test_datasets_token_file.cpp
        tests/test_datasets_tokens_section.cpp
        tests/test_datasets_window_layout.cpp)
