# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# failure_expectations: assertions over a failed Status and a thrown Error.

sushiai_add_internal_module(NAME failure_expectations TIER core
    DEPENDS core
    EXTERNAL_PUBLIC GTest::gtest)
