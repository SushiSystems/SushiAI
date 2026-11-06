# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# umbrella: the header that includes SushiAI and every header of language.

sushiai_add_module(NAME umbrella TIER language
    PUBLIC_DEPENDS
        language/blocks
        language/datasets
        language/generation
        language/registration
        language/text
        language/training
        umbrella)
