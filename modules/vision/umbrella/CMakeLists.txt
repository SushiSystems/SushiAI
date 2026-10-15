# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# umbrella: the header that includes SushiAI and every header of vision.

sushiai_add_module(NAME umbrella TIER vision
    PUBLIC_DEPENDS
        umbrella
        vision/blocks
        vision/codecs
        vision/datasets
        vision/detect
        vision/imaging
        vision/importers
        vision/registration
        vision/segment
        vision/training)
