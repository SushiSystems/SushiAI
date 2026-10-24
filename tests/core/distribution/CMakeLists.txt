# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module distribution; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER distribution
    SOURCES
        test_cluster_section.cpp
        test_exact_sum.cpp
        test_gradient_exchange.cpp
        test_host_access.cpp
        test_parameter_publish.cpp
        test_replica_draws.cpp
        test_replica_kernel.cpp
        test_replica_model.cpp
        test_replica_plan.cpp
        test_replica_position.cpp
        test_replica_step.cpp
        test_replica_task.cpp
    DEPENDS
        allocation
        autograd
        config
        execution
        failure_expectations
        graph)
