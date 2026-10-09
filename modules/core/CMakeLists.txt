# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# core: the tier that knows no layer type and no domain.
# A module is added by its folder and one line here.

add_subdirectory(core)
add_subdirectory(version)
add_subdirectory(graph)
add_subdirectory(config)
add_subdirectory(autograd)
add_subdirectory(optim)
add_subdirectory(data)
add_subdirectory(eager)
