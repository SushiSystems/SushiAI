# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module data; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER data
    SOURCES
        test_dataset_registry.cpp
        test_epoch_permutation.cpp
        test_target.cpp
        test_threaded_loader.cpp
    DEPENDS
        failure_expectations)
