# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# The test program of the module vision/datasets; see cmake/SushiAITestProgram.cmake.

sushiai_add_test_program(OWNER vision/datasets
    SOURCES
        test_augment_pipeline.cpp
        test_batch_inputs.cpp
        test_binary_collator.cpp
        test_cifar.cpp
        test_classification_collator.cpp
        test_coco_json.cpp
        test_collator.cpp
        test_csv.cpp
        test_detection_collator.cpp
        test_fit_mask.cpp
        test_fit_pipeline.cpp
        test_fit_sample.cpp
        test_fitted_sample.cpp
        test_idx.cpp
        test_idx_file.cpp
        test_image_extensions.cpp
        test_image_folder.cpp
        test_load_sample_image.cpp
        test_load_sample_mask.cpp
        test_mask_png.cpp
        test_mnist.cpp
        test_multi_label_collator.cpp
        test_pipeline_table.cpp
        test_polygon_mask.cpp
        test_reader_dataset.cpp
        test_reader_section.cpp
        test_reader_table.cpp
        test_regression_collator.cpp
        test_sample.cpp
        test_sample_cache.cpp
        test_sample_pipeline.cpp
        test_segmentation_collator.cpp
        test_stored_pixels.cpp
        test_table_dataset.cpp
        test_task_table.cpp
        test_value_table.cpp
        test_voc_xml.cpp
        test_xml_element.cpp
        test_yolo_txt.cpp
    DEPENDS
        failure_expectations
        model_text)
