# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# model_text: the JSON text of a test parsed into config values.

sushiai_add_internal_module(NAME model_text TIER core
    DEPENDS config)
