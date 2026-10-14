# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module io; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER io
    SOURCES
        test_checkpoint.cpp
        test_checkpoint_format.cpp
        test_import_weights.cpp
        test_imported_model_writer.cpp
        test_importer_registry.cpp
        test_run_checkpoint.cpp
        test_run_progress.cpp
    DEPENDS
        classifier_objective
        failure_expectations
        layer_rows
        model_text
        scratch_files
        tensor_values)
