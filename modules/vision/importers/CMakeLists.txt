# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# importers: the readers of other frameworks' model and weight files.

sushiai_add_module(NAME importers TIER vision
    SOURCES
        source/caffe_import.cpp
        source/caffe_op_batch_norm.cpp
        source/caffe_op_concat.cpp
        source/caffe_op_convolution.cpp
        source/caffe_op_dropout.cpp
        source/caffe_op_eltwise.cpp
        source/caffe_op_inner_product.cpp
        source/caffe_op_input.cpp
        source/caffe_op_pooling.cpp
        source/caffe_op_relu.cpp
        source/caffe_op_softmax.cpp
        source/caffe_operator_table.cpp
        source/caffe_parameters.cpp
        source/darknet_config.cpp
        source/darknet_fold.cpp
        source/darknet_import.cpp
        source/darknet_layer_convolutional.cpp
        source/darknet_layer_maxpool.cpp
        source/darknet_layer_route.cpp
        source/darknet_layer_shortcut.cpp
        source/darknet_layer_table.cpp
        source/darknet_layer_upsample.cpp
        source/darknet_layer_yolo.cpp
        source/darknet_model.cpp
        source/darknet_recipe.cpp
        source/darknet_store.cpp
        source/darknet_tensors.cpp
        source/darknet_training.cpp
        source/darknet_weights.cpp
        source/onnx_fusion.cpp
        source/onnx_import.cpp
        source/onnx_op_add.cpp
        source/onnx_op_batch_normalization.cpp
        source/onnx_op_concat.cpp
        source/onnx_op_constant.cpp
        source/onnx_op_conv.cpp
        source/onnx_op_div.cpp
        source/onnx_op_flatten.cpp
        source/onnx_op_gemm.cpp
        source/onnx_op_identity.cpp
        source/onnx_op_mat_mul.cpp
        source/onnx_op_max_pool.cpp
        source/onnx_op_mul.cpp
        source/onnx_op_relu.cpp
        source/onnx_op_reshape.cpp
        source/onnx_op_resize.cpp
        source/onnx_op_sigmoid.cpp
        source/onnx_op_slice.cpp
        source/onnx_op_softmax.cpp
        source/onnx_op_split.cpp
        source/onnx_op_sub.cpp
        source/onnx_op_transpose.cpp
        source/onnx_operator_table.cpp
        source/onnx_translation.cpp
        source/ultralytics_checkpoint.cpp
        source/ultralytics_import.cpp
        source/ultralytics_model.cpp
        source/ultralytics_module_c2f.cpp
        source/ultralytics_module_c2psa.cpp
        source/ultralytics_module_c3.cpp
        source/ultralytics_module_c3k2.cpp
        source/ultralytics_module_concat.cpp
        source/ultralytics_module_conv.cpp
        source/ultralytics_module_detect.cpp
        source/ultralytics_module_sppf.cpp
        source/ultralytics_module_table.cpp
        source/ultralytics_module_upsample.cpp
        source/ultralytics_tensors.cpp
    PUBLIC_DEPENDS config failure interchange io nn shape vision/detect
    PRIVATE_DEPENDS dtype optim seam
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
