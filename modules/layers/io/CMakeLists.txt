# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# io: checkpoints and the import of weights into a model state.

sushiai_add_module(NAME io TIER layers
    SOURCES
        source/checkpoint.cpp
        source/checkpoint_format.cpp
        source/import_weights.cpp
        source/imported_model_writer.cpp
        source/importer_registry.cpp
        source/model_records.cpp
        source/run_checkpoint.cpp
        source/run_progress.cpp
    PUBLIC_DEPENDS config dtype failure nn optim shape
    PRIVATE_DEPENDS execution seam
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
