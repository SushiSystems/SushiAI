# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# layers: the tier of the layers and of what trains, saves and loads a model of them.
# A module is added by its folder and one line here.

add_subdirectory(nn)
add_subdirectory(io)
add_subdirectory(train)
add_subdirectory(umbrella)
