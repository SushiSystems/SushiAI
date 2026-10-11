# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# umbrella: the header that includes SushiAI and every header of language.

sushiai_add_module(NAME umbrella TIER language
    PUBLIC_DEPENDS umbrella language/text language/blocks language/datasets
                   language/training language/generation language/registration)
