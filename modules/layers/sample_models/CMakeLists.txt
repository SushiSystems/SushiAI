# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# sample_models: small models the tests build.

sushiai_add_module(NAME sample_models TIER layers INTERNAL
    PUBLIC_DEPENDS failure nn shape)
