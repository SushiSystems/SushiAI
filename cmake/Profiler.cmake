# Profiler.cmake
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# SushiProfiler, the stack's one profiler, exposed as SushiProfiler::SushiProfiler. The CLI
# resolves its pin in cli/sushistack.deps.toml and hands SushiProfiler_DIR; only
# modules/core/profile and the modules that name it link it, privately.
option(SA_ENABLE_PROFILER "Record SushiAI's zones and periods through SushiProfiler" ON)
set(SA_PROFILE_LEVEL 2 CACHE STRING "Compile-time ceiling of the SA_ZONE family; 0 removes every site")

# What a module that names `profile` links privately: SushiProfiler, or nothing when it is off.
set(SA_PROFILER_LINKS "")
if(SA_ENABLE_PROFILER)
    if(NOT TARGET SushiProfiler::SushiProfiler)
        find_package(SushiProfiler CONFIG REQUIRED)
    endif()
    set(SA_PROFILER_LINKS SushiProfiler::SushiProfiler)
endif()
