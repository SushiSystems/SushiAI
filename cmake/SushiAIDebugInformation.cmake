# SushiAIDebugInformation.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAIDebugInformation.cmake - how much debug information a Debug tree carries, as a target.
#
# Single responsibility: define an INTERFACE target `sushiai_debug_info` carrying the host debug
# information of the Debug configuration. Debug is the development build: host code unoptimised,
# with line tables only, which give a stack trace its files and lines at a fraction of full debug
# information's compile memory, object size and link time. CMake states -g for Debug and the
# later -gline-tables-only overrides it. RelWithDebInfo keeps full information on optimised code.
# MSVC's own frontend has no such level and keeps what CMake states.
#
# SushiAIBuildPolicy.cmake applies this PRIVATE, like the warning and sanitizer targets.

add_library(sushiai_debug_info INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(sushiai_debug_info INTERFACE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-gline-tables-only>")
endif()
