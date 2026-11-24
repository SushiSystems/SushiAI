# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# interchange: the tables of foreign graph formats (ONNX, Caffe) and the rewrites they share.

sushiai_add_module(NAME interchange TIER layers
    SOURCES
        source/axis_translation.cpp
        source/caffe_net.cpp
        source/caffe_v1_types.cpp
        source/caffe_writer.cpp
        source/constant_folding.cpp
        source/onnx_graph.cpp
        source/onnx_writer.cpp
        source/tensor_bytes.cpp
        source/wire_values.cpp
    PUBLIC_DEPENDS failure io
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
