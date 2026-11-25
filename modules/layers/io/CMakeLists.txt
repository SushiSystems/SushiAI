# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# io: checkpoints and the import of weights into a model state.

sushiai_add_module(NAME io TIER layers
    SOURCES
        source/checkpoint.cpp
        source/checkpoint_format.cpp
        source/committed_file.cpp
        source/exported_model_writer.cpp
        source/exporter_registry.cpp
        source/filter_layout.cpp
        source/import_weights.cpp
        source/imported_model_writer.cpp
        source/importer_registry.cpp
        source/inflate.cpp
        source/model_files.cpp
        source/model_records.cpp
        source/name_map.cpp
        source/npy_reader.cpp
        source/npz_weight_source.cpp
        source/pickle_data.cpp
        source/pickle_reader.cpp
        source/pickle_writer.cpp
        source/protobuf_text.cpp
        source/protobuf_text_writer.cpp
        source/protobuf_wire.cpp
        source/protobuf_wire_writer.cpp
        source/run_checkpoint.cpp
        source/run_progress.cpp
        source/safetensors_reader.cpp
        source/safetensors_weight_source.cpp
        source/safetensors_writer.cpp
        source/torch_archive.cpp
        source/torch_weight_source.cpp
        source/torch_writer.cpp
        source/weight_import.cpp
        source/weight_source.cpp
        source/weight_export.cpp
        source/zip_archive.cpp
        source/zip_writer.cpp
    PUBLIC_DEPENDS config dtype failure nn optim shape
    PRIVATE_DEPENDS allocation seam
    EXTERNAL_PUBLIC SushiRuntime::core
    EXTERNAL_PRIVATE_LINKS SushiLog::text)
