# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# reference_files: the YOLO26 reference tensors and the comparisons against them.

sushiai_add_module(NAME reference_files TIER layers INTERNAL
    PUBLIC_DEPENDS dtype failure io shape
    EXTERNAL_PUBLIC GTest::gtest SushiRuntime::core)
