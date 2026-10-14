# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# reference_files: the YOLO26 reference tensors and the comparisons against them.

sushiai_add_internal_module(NAME reference_files TIER layers
    DEPENDS io
    EXTERNAL_PUBLIC GTest::gtest)
