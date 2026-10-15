# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# nn: the layers, the model graph, the model state and the node registry.

sushiai_add_module(NAME nn TIER layers
    SOURCES
        source/add.cpp
        source/avg_pool2d.cpp
        source/batch_norm2d.cpp
        source/built_rows.cpp
        source/channel_slice.cpp
        source/concat.cpp
        source/conv2d.cpp
        source/depthwise_conv2d.cpp
        source/dropout.cpp
        source/embedding.cpp
        source/flatten.cpp
        source/global_avg_pool.cpp
        source/identity.cpp
        source/init.cpp
        source/layer_node.cpp
        source/layer_norm.cpp
        source/layer_registry.cpp
        source/layer_stack.cpp
        source/layers.cpp
        source/leaky_relu.cpp
        source/max_pool2d.cpp
        source/mish.cpp
        source/model_graph.cpp
        source/model_outputs.cpp
        source/model_state.cpp
        source/node_attributes.cpp
        source/node_registry.cpp
        source/silu.cpp
        source/trace.cpp
        source/unflatten.cpp
        source/upsample2d.cpp
    PUBLIC_DEPENDS config core graph)
