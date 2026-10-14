# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# tensor_values: host reads and writes of the elements of a tensor.

sushiai_add_internal_module(NAME tensor_values TIER core
    DEPENDS core
    EXTERNAL_PUBLIC GTest::gtest)
