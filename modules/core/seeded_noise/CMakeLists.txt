# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# seeded_noise: a reproducible stream of values from a seed.

sushiai_add_module(NAME seeded_noise TIER core INTERNAL
    EXTERNAL_PUBLIC SushiRuntime::core)
