# SushiAICompilerWarnings.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026 Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiAICompilerWarnings.cmake — the project's warning policy as a target.
#
# Single responsibility: define an INTERFACE target `sushiai_warnings`
# carrying the compiler warning level and the preprocessor definitions a
# warning-clean build needs. Architecture targeting lives in
# SushiAIOptimization.cmake because it answers to a different owner: warnings
# are a review standard, architecture is a deployment choice.
#
# SushiAIBuildPolicy.cmake applies this PRIVATE, so `-Werror` is a rule for
# this tree and never something an installed consumer inherits — mirrors
# sushiblas/cmake/SushiBLASCompilerWarnings.cmake exactly.

add_library(sushiai_warnings INTERFACE)

if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    # Windows / MSVC / Intel icx-cl
    target_compile_options(sushiai_warnings INTERFACE
        /W4 /permissive-
    )
    target_compile_definitions(sushiai_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
    # Linux / GCC / Clang / Intel oneAPI
    target_compile_options(sushiai_warnings INTERFACE
        -Wall
        -Wextra
        -Werror
        -pedantic
    )
endif()
