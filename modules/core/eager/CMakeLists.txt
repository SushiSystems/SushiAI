# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# eager: the host-lane units of the eager front end.

sushiai_add_module(NAME eager TIER core
    SOURCES
        source/region_signature.cpp
        source/scope_partitioner.cpp
        source/scope_trace.cpp
    PUBLIC_DEPENDS core graph autograd
    TESTS
        tests/test_region_signature.cpp
        tests/test_scope_partitioner.cpp
        tests/test_scope_trace.cpp)
