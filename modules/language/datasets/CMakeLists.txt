# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

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
    PUBLIC_DEPENDS config data dtype failure shape language/text
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
