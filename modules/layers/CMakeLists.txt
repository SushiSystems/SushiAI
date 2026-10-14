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

# The modules only the tests use; see cmake/SushiAITestProgram.cmake.
if(SA_BUILD_TESTS)
    add_subdirectory(layer_rows)
    add_subdirectory(layer_traces)
    add_subdirectory(sample_models)
    add_subdirectory(reference_files)
    add_subdirectory(classifier_objective)
endif()
