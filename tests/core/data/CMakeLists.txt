# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module data; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER data
    SOURCES
        compile_input.cpp
        compile_roles.cpp
        test_batch_allocator.cpp
        test_dataset.cpp
        test_dataset_registry.cpp
        test_epoch.cpp
        test_epoch_permutation.cpp
        test_in_memory_dataset.cpp
        test_in_memory_epoch_aware.cpp
        test_materialised_dataset.cpp
        test_target.cpp
        test_threaded_loader.cpp
    DEPENDS
        failure_expectations
        model_text
        tensor_values)
