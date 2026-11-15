# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# importers: the readers of other frameworks' model and weight files.

sushiai_add_module(NAME importers TIER vision
    SOURCES
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
        source/ultralytics_checkpoint.cpp
        source/ultralytics_import.cpp
        source/ultralytics_model.cpp
        source/ultralytics_module_c2psa.cpp
        source/ultralytics_module_c3k2.cpp
        source/ultralytics_module_concat.cpp
        source/ultralytics_module_conv.cpp
        source/ultralytics_module_detect.cpp
        source/ultralytics_module_sppf.cpp
        source/ultralytics_module_table.cpp
        source/ultralytics_module_upsample.cpp
        source/ultralytics_tensors.cpp
    PUBLIC_DEPENDS config failure io nn shape vision/detect
    PRIVATE_DEPENDS dtype optim seam
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
