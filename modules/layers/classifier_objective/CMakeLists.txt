# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# classifier_objective: the objective of a graph build_classifier() traced.

sushiai_add_internal_module(NAME classifier_objective TIER layers
    DEPENDS train)
