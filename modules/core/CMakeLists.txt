# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# core: the tier that knows no layer type and no domain.
# A module is added by its folder and one line here.

add_subdirectory(log)
add_subdirectory(failure)
add_subdirectory(dtype)
add_subdirectory(shape)
add_subdirectory(ops)
add_subdirectory(seam)
add_subdirectory(execution)
add_subdirectory(allocation)
add_subdirectory(version)
add_subdirectory(graph)
add_subdirectory(config)
add_subdirectory(autograd)
add_subdirectory(optim)
add_subdirectory(data)
add_subdirectory(eager)

# The modules only the tests use; see cmake/SushiAITestProgram.cmake.
if(SA_BUILD_TESTS)
    add_subdirectory(engine_fixture)
    add_subdirectory(failure_expectations)
    add_subdirectory(tensor_values)
    add_subdirectory(seeded_noise)
    add_subdirectory(scratch_files)
    add_subdirectory(model_text)
    add_subdirectory(darknet_reference)
endif()
