# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# failure_expectations: assertions over a failed Status and a thrown Error.

sushiai_add_module(NAME failure_expectations TIER core INTERNAL
    PUBLIC_DEPENDS failure
    EXTERNAL_PUBLIC GTest::gtest)
