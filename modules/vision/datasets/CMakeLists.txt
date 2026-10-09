# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# datasets: the datasets of computer vision.

sushiai_add_module(NAME datasets TIER vision
    SOURCES
        source/mnist.cpp
    PUBLIC_DEPENDS core config data)
