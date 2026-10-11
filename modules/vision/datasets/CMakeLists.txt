# CMakeLists.txt
# SushiAI - https://github.com/SushiSystems/SushiAI
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.

# datasets: the datasets of computer vision.

sushiai_add_module(NAME datasets TIER vision
    SOURCES
        source/augment_pipeline.cpp
        source/batch_inputs.cpp
        source/binary_collator.cpp
        source/cifar.cpp
        source/classification_collator.cpp
        source/coco_json.cpp
        source/collate_support.cpp
        source/csv.cpp
        source/detection_collator.cpp
        source/disk.cpp
        source/fit_mask.cpp
        source/fit_pipeline.cpp
        source/fit_sample.cpp
        source/idx.cpp
        source/idx_file.cpp
        source/image_extensions.cpp
        source/image_folder.cpp
        source/load_sample_image.cpp
        source/load_sample_mask.cpp
        source/mask_png.cpp
        source/mnist.cpp
        source/multi_label_collator.cpp
        source/pipeline_table.cpp
        source/polygon_mask.cpp
        source/reader_dataset.cpp
        source/reader_section.cpp
        source/reader_table.cpp
        source/regression_collator.cpp
        source/sample_cache.cpp
        source/section_count.cpp
        source/segmentation_collator.cpp
        source/stored_pixels.cpp
        source/table_dataset.cpp
        source/task_dataset.cpp
        source/task_table.cpp
        source/voc_xml.cpp
        source/xml_element.cpp
        source/yolo_txt.cpp
    PUBLIC_DEPENDS core config data vision/imaging vision/codecs
    PRIVATE_DEPENDS vision/segment
    TESTS
        tests/test_datasets_cifar.cpp
        tests/test_datasets_coco_json.cpp
        tests/test_datasets_collators.cpp
        tests/test_datasets_csv.cpp
        tests/test_datasets_fit_mask.cpp
        tests/test_datasets_fit_sample.cpp
        tests/test_datasets_idx.cpp
        tests/test_datasets_idx_file.cpp
        tests/test_datasets_image_folder.cpp
        tests/test_datasets_load_sample_image.cpp
        tests/test_datasets_mask_png.cpp
        tests/test_datasets_pipelines.cpp
        tests/test_datasets_polygon_mask.cpp
        tests/test_datasets_reader_dataset.cpp
        tests/test_datasets_reader_section.cpp
        tests/test_datasets_sample_cache.cpp
        tests/test_datasets_stored_pixels.cpp
        tests/test_datasets_table_dataset.cpp
        tests/test_datasets_task_collators.cpp
        tests/test_datasets_voc_xml.cpp
        tests/test_datasets_xml_element.cpp
        tests/test_datasets_yolo_txt.cpp)
