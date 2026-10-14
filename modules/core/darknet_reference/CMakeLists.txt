# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# darknet_reference: the readers of the files a Darknet run dumped.

sushiai_add_internal_module(NAME darknet_reference TIER core
    SOURCES
        source/darknet_reference.cpp
    DEPENDS config
    EXTERNAL_PUBLIC GTest::gtest)
