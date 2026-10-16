# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module nn; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER nn
    SOURCES
        test_add.cpp
        test_avg_pool2d.cpp
        test_batch_norm2d.cpp
        test_buffer.cpp
        test_built_rows.cpp
        test_channel_slice.cpp
        test_concat.cpp
        test_conv2d.cpp
        test_depthwise_conv2d.cpp
        test_dropout.cpp
        test_embedding.cpp
        test_flatten.cpp
        test_global_avg_pool.cpp
        test_host_table.cpp
        test_identity.cpp
        test_init.cpp
        test_layer.cpp
        test_layer_node.cpp
        test_layer_norm.cpp
        test_layer_registry.cpp
        test_layer_stack.cpp
        test_layers.cpp
        test_leaky_relu.cpp
        test_loss.cpp
        test_max_pool2d.cpp
        test_mish.cpp
        test_model_graph.cpp
        test_model_outputs.cpp
        test_model_state.cpp
        test_module.cpp
        test_module_list.cpp
        test_module_node.cpp
        test_node.cpp
        test_node_attributes.cpp
        test_node_registry.cpp
        test_silu.cpp
        test_throwing_bridge.cpp
        test_trace.cpp
        test_unflatten.cpp
        test_upsample2d.cpp
    DEPENDS
        failure_expectations
        layer_rows
        layer_traces
        model_text
        reference_files
        seeded_noise
        tensor_values)
