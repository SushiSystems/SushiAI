# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# layer_traces: the operations, outputs and state one training trace leaves.

sushiai_add_module(NAME layer_traces TIER layers INTERNAL
    PUBLIC_DEPENDS core graph nn)
