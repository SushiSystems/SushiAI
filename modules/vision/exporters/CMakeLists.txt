# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# exporters: the writers of other frameworks' model and weight files.

sushiai_add_module(NAME exporters TIER vision
    SOURCES
        source/caffe_export.cpp
        source/caffe_export_progress.cpp
        source/caffe_row_add.cpp
        source/caffe_row_avgpool2d.cpp
        source/caffe_row_batchnorm2d.cpp
        source/caffe_row_concat.cpp
        source/caffe_row_conv2d.cpp
        source/caffe_row_dropout.cpp
        source/caffe_row_flatten.cpp
        source/caffe_row_global_avgpool.cpp
        source/caffe_row_leaky_relu.cpp
        source/caffe_row_linear.cpp
        source/caffe_row_maximum.cpp
        source/caffe_row_maxpool2d.cpp
        source/caffe_row_mul.cpp
        source/caffe_row_relu.cpp
        source/caffe_row_softmax.cpp
        source/caffe_row_table.cpp
        source/darknet_export.cpp
        source/darknet_export_progress.cpp
        source/darknet_files.cpp
        source/darknet_head_sections.cpp
        source/darknet_row_add.cpp
        source/darknet_row_concat.cpp
        source/darknet_row_conv2d.cpp
        source/darknet_row_maxpool2d.cpp
        source/darknet_row_slice_channels.cpp
        source/darknet_row_table.cpp
        source/darknet_row_upsample.cpp
        source/darknet_training_lines.cpp
        source/export_walk.cpp
        source/onnx_export.cpp
        source/onnx_export_progress.cpp
        source/onnx_row_add.cpp
        source/onnx_row_batchnorm2d.cpp
        source/onnx_row_concat.cpp
        source/onnx_row_conv2d.cpp
        source/onnx_row_div.cpp
        source/onnx_row_flatten.cpp
        source/onnx_row_linear.cpp
        source/onnx_row_maxpool2d.cpp
        source/onnx_row_mul.cpp
        source/onnx_row_relu.cpp
        source/onnx_row_sigmoid.cpp
        source/onnx_row_silu.cpp
        source/onnx_row_slice_channels.cpp
        source/onnx_row_softmax.cpp
        source/onnx_row_sub.cpp
        source/onnx_row_table.cpp
        source/onnx_row_transpose.cpp
        source/onnx_row_upsample.cpp
        source/row_keys.cpp
        source/row_weights.cpp
    PUBLIC_DEPENDS failure io interchange
    PRIVATE_DEPENDS config dtype optim vision/detect
    EXTERNAL_PUBLIC SushiRuntime::core)
