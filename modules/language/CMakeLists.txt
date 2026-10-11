# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# language: the language domain; it may read core and layers and no other domain.
# A module is added by its folder and one line here.

add_subdirectory(text)
add_subdirectory(blocks)
add_subdirectory(datasets)
add_subdirectory(training)
add_subdirectory(generation)
add_subdirectory(registration)
add_subdirectory(umbrella)
