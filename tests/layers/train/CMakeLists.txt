# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module train; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER train
    SOURCES
        test_batch_role.cpp
        test_binary_confusion.cpp
        test_binary_cross_entropy_objective.cpp
        test_buffer_update_recorder.cpp
        test_decision_threshold.cpp
        test_dense_losses.cpp
        test_dense_targets.cpp
        test_evaluation_step.cpp
        test_feed.cpp
        test_host_values.cpp
        test_input_match.cpp
        test_mean_squared_error_objective.cpp
        test_metrics.cpp
        test_multi_label_objective.cpp
        test_objective.cpp
        test_objective_context.cpp
        test_objective_registry.cpp
        test_one_hot.cpp
        test_parameter_groups.cpp
        test_role_match.cpp
        test_step_view.cpp
        test_steps.cpp
        test_sum_objective.cpp
        test_trainer/test_compile_once.cpp
        test_trainer/test_convergence.cpp
        test_trainer/test_dense_objectives.cpp
        test_trainer/test_determinism.cpp
        test_trainer/test_dropout.cpp
        test_trainer/test_fused_optimizer.cpp
        test_trainer/test_fusion_numerics.cpp
        test_trainer/test_learning_curve.cpp
        test_trainer/test_loss_scaling.cpp
        test_trainer/test_mixed_precision.cpp
        test_trainer/test_resilience.cpp
        test_trainer/test_static_memory.cpp
        test_trainer/test_token_feed.cpp
        test_training_graph.cpp
        test_training_step.cpp
    DEPENDS
        classifier_objective
        failure_expectations
        sample_models
        seeded_noise
        tensor_values)
