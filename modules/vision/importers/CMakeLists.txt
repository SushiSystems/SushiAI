# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

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
    PUBLIC_DEPENDS core config optim nn io vision/detect
    EXTERNAL_PUBLIC SushiRuntime::SushiRuntime)
