# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# layer_rows: the rows of the layer registry a test builds.

sushiai_add_internal_module(NAME layer_rows TIER layers
    DEPENDS nn model_text failure_expectations)
