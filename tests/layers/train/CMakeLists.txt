# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module train; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER train
    SOURCES
        test_binary_confusion.cpp
        test_binary_cross_entropy_objective.cpp
        test_dense_losses.cpp
        test_dense_targets.cpp
        test_evaluation_step.cpp
        test_input_match.cpp
        test_mean_squared_error_objective.cpp
        test_metrics.cpp
        test_multi_label_objective.cpp
        test_objective.cpp
        test_objective_context.cpp
        test_objective_registry.cpp
        test_one_hot.cpp
        test_role_match.cpp
        test_steps.cpp
        test_sum_objective.cpp
    DEPENDS
        failure_expectations
        sample_models
        tensor_values)
