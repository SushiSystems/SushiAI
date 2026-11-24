# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under the Apache License 2.0. See LICENSE.

# The test program of the module io; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER io
    SOURCES
        test_checkpoint.cpp
        test_checkpoint_format.cpp
        test_exported_model.cpp
        test_exported_model_writer.cpp
        test_exporter_registry.cpp
        test_filter_layout.cpp
        test_import_weights.cpp
        test_in_memory_importer.cpp
        test_imported_model.cpp
        test_imported_model_writer.cpp
        test_importer_registry.cpp
        test_inflate.cpp
        test_memory_input.cpp
        test_model_files.cpp
        test_name_map.cpp
        test_npy_reader.cpp
        test_npz_weight_source.cpp
        test_pickle_data.cpp
        test_pickle_reader.cpp
        test_pickle_writer.cpp
        test_protobuf_text.cpp
        test_protobuf_text_writer.cpp
        test_protobuf_wire.cpp
        test_protobuf_wire_writer.cpp
        test_run_checkpoint.cpp
        test_run_progress.cpp
        test_safetensors_reader.cpp
        test_safetensors_weight_source.cpp
        test_safetensors_writer.cpp
        test_torch_archive.cpp
        test_torch_weight_source.cpp
        test_torch_writer.cpp
        test_weight_import.cpp
        test_weight_source.cpp
        test_weight_export.cpp
        test_zip_archive.cpp
        test_zip_writer.cpp
    DEPENDS
        classifier_objective
        failure_expectations
        layer_rows
        model_text
        scratch_files
        tensor_values)
