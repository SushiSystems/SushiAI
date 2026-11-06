# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# classifier_objective: the objective of a graph build_classifier() traced.

sushiai_add_module(NAME classifier_objective TIER layers INTERNAL
    PUBLIC_DEPENDS graph train)
