# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test programs of the core tier, one per module.

add_subdirectory(autograd)
add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(darknet_reference)
add_subdirectory(data)
add_subdirectory(eager)
add_subdirectory(engine_fixture)
add_subdirectory(failure_expectations)
add_subdirectory(graph)
add_subdirectory(model_text)
add_subdirectory(optim)
add_subdirectory(scratch_files)
add_subdirectory(seeded_noise)
add_subdirectory(tensor_values)
add_subdirectory(version)
