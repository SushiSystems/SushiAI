# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# config: the JSON value type, the model and training documents and their writers.

# nlohmann_json is named by source/value.cpp alone and reaches no header, so
# it is taken as include directories and stays out of the package: see
# sushiai_add_module() and tests/package/.
find_package(nlohmann_json CONFIG REQUIRED)

sushiai_add_module(NAME config TIER core
    SOURCES
        source/json_events.cpp
        source/json_writer.cpp
        source/model_config.cpp
        source/model_writer.cpp
        source/section_reader.cpp
        source/train_config.cpp
        source/value.cpp
    PUBLIC_DEPENDS core
    EXTERNAL_PRIVATE_INCLUDES nlohmann_json::nlohmann_json
    TESTS
        tests/test_json_events.cpp
        tests/test_json_writer.cpp
        tests/test_model_writer.cpp
        tests/test_section_reader.cpp
        tests/test_train_config.cpp)
