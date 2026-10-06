# SushiAIOptimization.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAIOptimization.cmake — architecture targeting as a target.
#
# Single responsibility: define an INTERFACE target `sushiai_perf` carrying
# the code-generation flags that are a property of the machine a binary is
# built *for*. The optimization level itself is not here: CMake's own
# CMAKE_CXX_FLAGS_<CONFIG> already supplies -O3 / /O2 in Release.
#
# Architecture targeting is opt-in. `-march=native`/`/arch:AVX2` bakes the
# build machine's exact ISA into the binary, which is right for a developer
# benchmarking on the box in front of them and wrong for anything shipped.
# SushiAIBuildPolicy.cmake applies this PRIVATE, so it applies to this tree
# only and never reaches an installed consumer — mirrors
# sushiblas/cmake/SushiBLASOptimization.cmake exactly.

option(SA_NATIVE_ARCH
    "Build for the build machine's exact ISA (-march=native / /arch:AVX2). Developer convenience; must stay OFF for any binary that leaves this machine."
    OFF)

add_library(sushiai_perf INTERFACE)

if(SA_NATIVE_ARCH)
    if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
        target_compile_options(sushiai_perf INTERFACE /arch:AVX2)
    else()
        target_compile_options(sushiai_perf INTERFACE -march=native)
    endif()
    message(STATUS "SA_NATIVE_ARCH=ON — binaries target this machine's ISA and are not portable.")
endif()
