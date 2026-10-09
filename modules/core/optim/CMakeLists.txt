# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# optim: the optimizers, the loss scaler and the scalars a step reads.

sushiai_add_module(NAME optim TIER core
    SOURCES
        source/adamw.cpp
        source/loss_scaler.cpp
        source/optimizer.cpp
        source/optimizer_registry.cpp
        source/sgd.cpp
        source/step_scalars.cpp
    PUBLIC_DEPENDS core graph config)
