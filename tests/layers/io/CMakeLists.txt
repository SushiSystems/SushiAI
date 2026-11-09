# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module io; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER io
    SOURCES
        test_checkpoint.cpp
        test_checkpoint_format.cpp
        test_import_weights.cpp
        test_in_memory_importer.cpp
        test_imported_model.cpp
        test_imported_model_writer.cpp
        test_importer_registry.cpp
        test_memory_input.cpp
        test_run_checkpoint.cpp
        test_run_progress.cpp
    DEPENDS
        classifier_objective
        failure_expectations
        layer_rows
        model_text
        scratch_files
        tensor_values)
