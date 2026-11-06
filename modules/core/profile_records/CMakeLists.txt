# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# profile_records: test-only helpers that write a profiling configuration and read a record back.

sushiai_add_module(NAME profile_records TIER core INTERNAL
    PUBLIC_DEPENDS profile
    EXTERNAL_PUBLIC ${SA_PROFILER_LINKS})
