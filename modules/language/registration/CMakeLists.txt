# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# registration: the functions that add the rows of language to a registry.

sushiai_add_module(NAME registration TIER language
    SOURCES
        source/registration.cpp
    PUBLIC_DEPENDS failure
    PRIVATE_DEPENDS data nn train language/blocks language/datasets language/training)
