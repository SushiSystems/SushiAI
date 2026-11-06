# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module profile_records; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER profile_records
    SOURCES
        test_profile_records.cpp
    DEPENDS
        profile_records
        scratch_files
    LINKS
        ${SA_PROFILER_LINKS})
