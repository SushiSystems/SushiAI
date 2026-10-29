# SushiAIAbiPinning.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAIAbiPinning.cmake - the C runtime and iterator settings every target of this tree shares.
#
# Single responsibility: set, once at directory scope, the flags that decide whether two binaries
# can share C++ objects. They are the settings of SushiEngine's AbiPinning.cmake, for the same
# reason: the libraries below this repository arrive as Release packages.

# The runtime selection only takes effect under the NEW behaviour; refuse rather than pin nothing.
cmake_policy(GET CMP0091 sushiai_runtime_policy)
if(NOT sushiai_runtime_policy STREQUAL "NEW")
    message(FATAL_ERROR
        "SushiAIAbiPinning: policy CMP0091 is '${sushiai_runtime_policy}', so "
        "CMAKE_MSVC_RUNTIME_LIBRARY would be ignored. Raise cmake_minimum_required to 3.15 "
        "or later.")
endif()
unset(sushiai_runtime_policy)

# Every build type takes the release C runtime. Debug is the development build: unoptimised host
# code against the same packages as Release, so it needs no debug build of SushiRuntime or
# SushiBLAS, and its STL runs without iterator checks. Memory errors are the sanitizer tree's job.
# The rule is in the build-throughput skill of SushiSkills. Ignored off the MSVC ABI.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# A Debug consumer of an imported target links its Release build, which shares that CRT.
set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Release RelWithDebInfo MinSizeRel None "")

# The MSVC STL's container layout depends on this; libstdc++ and libc++ never read it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX>:_ITERATOR_DEBUG_LEVEL=0>")
endif()
