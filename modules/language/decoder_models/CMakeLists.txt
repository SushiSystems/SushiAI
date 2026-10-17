# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# decoder_models: the small decoder model files the language tests build.

sushiai_add_module(NAME decoder_models TIER language INTERNAL
    PUBLIC_DEPENDS config failure model_text nn language/registration)
