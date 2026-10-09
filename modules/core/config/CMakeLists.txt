# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# config: the JSON value type and the model and training documents.

# nlohmann_json is named by source/value.cpp alone and reaches no header, so
# it is taken as include directories and stays out of the package: see
# sushiai_add_module() and tests/package/.
find_package(nlohmann_json CONFIG REQUIRED)

sushiai_add_module(NAME config TIER core
    SOURCES
        source/model_config.cpp
        source/train_config.cpp
        source/value.cpp
    PUBLIC_DEPENDS core
    EXTERNAL_PRIVATE_INCLUDES nlohmann_json::nlohmann_json)
