# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# graph: the IR, its builder, the passes over it and the lowering to SushiBLAS.

sushiai_add_module(NAME graph TIER core
    SOURCES
        source/arena.cpp
        source/builder.cpp
        source/cast_insertion.cpp
        source/compiled_step.cpp
        source/fusion.cpp
        source/fusion_selector.cpp
        source/lowering.cpp
        source/lowering/activation.cpp
        source/lowering/context.cpp
        source/lowering/elementwise.cpp
        source/lowering/fused.cpp
        source/lowering/linalg.cpp
        source/lowering/loss.cpp
        source/lowering/norm.cpp
        source/lowering/shape.cpp
        source/lowering/spatial.cpp
        source/memory_plan.cpp
        source/precision.cpp
        source/rematerialise.cpp
        source/tensor_pool.cpp
    PUBLIC_DEPENDS core
    TESTS
        tests/test_batched_matmul.cpp
        tests/test_depthwise_ops.cpp
        tests/test_elementwise_ops.cpp
        tests/test_fusion.cpp
        tests/test_ir.cpp
        tests/test_lowering_table.cpp
        tests/test_memory_plan.cpp
        tests/test_step_recorder.cpp
        tests/test_traced.cpp)
