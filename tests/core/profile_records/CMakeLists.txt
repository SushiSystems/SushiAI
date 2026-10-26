# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module profile_records; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER profile_records
    SOURCES
        test_profile_records.cpp
    DEPENDS
        profile_records
        scratch_files
    LINKS
        ${SA_PROFILER_LINKS})
