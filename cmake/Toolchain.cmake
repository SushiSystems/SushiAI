# Toolchain.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# Toolchain — pick the default SYCL compiler before project()'s compiler probe.
#
# Mirrors the runtime's intel-llvm default: use clang++ when the caller named
# no compiler, so a plain configure lands on the primary vendor-neutral
# toolchain. The runtime, pulled in as a subproject, does the full toolchain
# selection; this only fixes the compiler the top-level probe validates. Pass
# -DCMAKE_CXX_COMPILER=... to override (e.g. acpp, icx-cl).
#
# Filename intentionally NOT SushiAI-prefixed — see cmake/Vcpkg.cmake's header
# comment; this file's logic is identically idempotent across sibling repos.
if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
    find_program(SA_CLANGXX NAMES clang++)
    if(SA_CLANGXX)
        set(CMAKE_CXX_COMPILER "${SA_CLANGXX}" CACHE FILEPATH "CXX compiler")
    endif()
endif()
