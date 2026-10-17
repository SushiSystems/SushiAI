# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# autograd: reverse-mode differentiation as a pass from IR to IR.

sushiai_add_module(NAME autograd TIER core
    SOURCES
        source/differentiator.cpp
        source/gradcheck.cpp
        source/rules/activation.cpp
        source/rules/elementwise.cpp
        source/rules/linalg.cpp
        source/rules/loss.cpp
        source/rules/norm.cpp
        source/rules/random.cpp
        source/rules/shape.cpp
        source/rules/spatial.cpp
        source/tape.cpp
    PUBLIC_DEPENDS failure graph ops
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
