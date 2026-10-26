# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# profile_records: test-only helpers that write a profiling configuration and read a record back.

sushiai_add_module(NAME profile_records TIER core INTERNAL
    EXTERNAL_PUBLIC ${SA_PROFILER_LINKS})
