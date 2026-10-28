# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module profile; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER profile
    SOURCES
        test_timelines.cpp
        test_profile.cpp
        test_timeline_binding.cpp
        test_zone_groups.cpp
    DEPENDS
        profile_records
    LINKS
        ${SA_PROFILER_LINKS}
    INCLUDES
        "${SA_PROFILE_GENERATED_INCLUDE}")
