# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module graph; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER graph
    SOURCES
        lowering/test_activation.cpp
        lowering/test_all.cpp
        lowering/test_context.cpp
        lowering/test_elementwise.cpp
        lowering/test_fused.cpp
        lowering/test_linalg.cpp
        lowering/test_loss.cpp
        lowering/test_norm.cpp
        lowering/test_random.cpp
        lowering/test_rule.cpp
        lowering/test_shape.cpp
        lowering/test_spatial.cpp
        lowering/test_unlowered.cpp
        test_arena.cpp
        test_builder.cpp
        test_cast_insertion.cpp
        test_compiled_step.cpp
        test_dropout_mask.cpp
        test_fusion.cpp
        test_fusion_selector.cpp
        test_ir.cpp
        test_lowering/test_hand_values.cpp
        test_lowering/test_host_values.cpp
        test_memory_plan.cpp
        test_node_range.cpp
        test_precision.cpp
        test_rematerialise.cpp
        test_replay_prelude.cpp
        test_state_id.cpp
        test_step_recorder.cpp
        test_step_recording.cpp
        test_tensor_pool.cpp
        test_traced.cpp
    DEPENDS
        failure_expectations
        tensor_values)
