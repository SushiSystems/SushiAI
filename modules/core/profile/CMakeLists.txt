# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# profile: the one module that includes SushiProfiler; the SA_ZONE family and the timelines.
# SA_ENABLE_PROFILER and SA_PROFILE_LEVEL reach the sources through the generated
# profile_config.hpp, never through compile definitions, so a change of level rebuilds only
# the files that include profile.hpp.

sushiai_add_module(NAME profile TIER core UNINSTALLED_HEADERS
    SOURCES
        source/timelines.cpp
    PRIVATE_DEPENDS failure
    EXTERNAL_PRIVATE_LINKS ${SA_PROFILER_LINKS})

if(SA_ENABLE_PROFILER)
    set(SA_PROFILE_ENABLED_VALUE 1)
else()
    set(SA_PROFILE_ENABLED_VALUE 0)
endif()

# The folder of the generated header; the programs and the tests that include profile.hpp add it.
set(profile_generated "${CMAKE_CURRENT_BINARY_DIR}/include")
set(SA_PROFILE_GENERATED_INCLUDE "${profile_generated}" CACHE INTERNAL "")
configure_file(profile_config.hpp.in
    "${profile_generated}/SushiAI/profile/profile_config.hpp" @ONLY)

sushiai_module_key(profile core profile_key)
sushiai_module_target("${profile_key}" profile_target)
target_include_directories(${profile_target} PUBLIC "$<BUILD_INTERFACE:${profile_generated}>")
