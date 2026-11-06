# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# tensor_values: host reads and writes of the elements of a tensor.

sushiai_add_module(NAME tensor_values TIER core INTERNAL
    PUBLIC_DEPENDS dtype
    EXTERNAL_PUBLIC GTest::gtest SushiRuntime::core)
