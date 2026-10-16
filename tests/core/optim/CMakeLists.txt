# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module optim; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER optim
    SOURCES
        test_adamw.cpp
        test_apply_step_schedule.cpp
        test_cosine_schedule.cpp
        test_cosine_schedule_section.cpp
        test_gradient_clip.cpp
        test_loss_scaler.cpp
        test_optimizer.cpp
        test_optimizer_registry.cpp
        test_rate_schedule.cpp
        test_rate_schedule_section.cpp
        test_run_schedule.cpp
        test_run_state.cpp
        test_sgd.cpp
        test_step_scalars.cpp
        test_step_schedule.cpp
        test_step_schedule_section.cpp
    DEPENDS
        failure_expectations
        tensor_values)
