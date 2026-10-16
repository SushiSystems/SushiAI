# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test programs of the vision tier, one per module.

add_subdirectory(blocks)
add_subdirectory(codecs)
add_subdirectory(darknet_forward_cases)
add_subdirectory(darknet_import_cases)
add_subdirectory(darknet_training_cases)
add_subdirectory(datasets)
add_subdirectory(detect)
add_subdirectory(imaging)
add_subdirectory(importers)
add_subdirectory(registration)
add_subdirectory(segment)
add_subdirectory(training)
add_subdirectory(umbrella)
