# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# layer_traces: the operations, outputs and state one training trace leaves.

sushiai_add_module(NAME layer_traces TIER layers INTERNAL
    PUBLIC_DEPENDS failure graph nn shape
    EXTERNAL_PUBLIC SushiRuntime::core)
