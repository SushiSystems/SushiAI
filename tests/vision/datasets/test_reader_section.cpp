/**************************************************************************/
/* test_reader_section.cpp                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/


/**
 * @file test_reader_section.cpp
 * @brief Tests the reader table, the task table and the dataset a data section builds of them.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/SushiAI.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides an engine and the sections these tests build datasets from. */
    class Unit_DatasetsReaderSection : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Parses a section that opens with a root and ends with @p members.
             * @param members JSON members, each led by a comma; may be empty.
             */
            Config::Value section_of(const Support::ScratchDirectory& scratch,
                                     std::string_view members) const
            {
                const std::string text = R"({"root": ")" + scratch.path().generic_string()
                                       + "\"" + std::string(members) + "}";
                return SaTest::attributes_from(text);
            }

            /** @brief Builds the dataset of a section called "data". */
            Result<std::unique_ptr<Data::Dataset>> build(const Config::Value& section)
            {
                return make_reader_dataset(*engine, section, "data");
            }

            /** @brief Writes two 2 by 4 white images; the second has one object of class 2. */
            void write_yolo_split(const Support::ScratchDirectory& scratch) const
            {
                for (const char* name : {"images/train/a.saim", "images/train/b.saim"})
                {
                    const std::filesystem::path file = scratch.path() / name;
                    std::filesystem::create_directories(file.parent_path());
                    Vision::Imaging::write_raw_image_file(
                        file, Support::uniform_image(ImageSize{2, 4}, 3, 255).view(),
                        Vision::Imaging::ChannelOrder::RGB);
                }

                scratch.write_text("labels/train/b.txt", "2 0.5 0.5 0.5 1.0\n");
            }

            /** @brief Writes one image file under each of two class folders of "train". */
            void write_folder_split(const Support::ScratchDirectory& scratch) const
            {
                scratch.write("train/cat/a.jpg", {1});
                scratch.write("train/dog/b.jpg", {1});
            }
    };

    /** @brief Returns the names of the rows of a table, in order. */
    template <typename Entry>
    std::vector<std::string_view> names_of(SushiRuntime::span<const Entry> rows)
    {
        std::vector<std::string_view> names;
        for (const Entry& row : rows)
            names.push_back(row.name);

        return names;
    }
} // namespace

/** @brief Tests that the reader table holds the eight readers of the module, by name. */
TEST(Unit_DatasetsReaderTable, ItListsTheEightReaders)
{
    const std::vector<std::string_view> expected = {"yolo_txt", "image_folder", "idx", "cifar",
                                                    "coco_json", "voc_xml", "mask_png", "csv"};

    EXPECT_EQ(names_of(reader_table()), expected);
    for (const ReaderEntry& row : reader_table())
    {
        EXPECT_NE(row.read, nullptr) << row.name;
        EXPECT_EQ(find_reader(row.name), &row) << row.name;
    }

    EXPECT_EQ(find_reader("tfrecord"), nullptr);
}

/** @brief Tests that a row lists the keys it reads of the section, and the first four none. */
TEST(Unit_DatasetsReaderTable, ARowListsTheKeysItReads)
{
    const auto keys_of = [](std::string_view name)
    {
        const ReaderEntry* const row = find_reader(name);
        return row == nullptr ? std::vector<std::string_view>{"no such row"}
                              : std::vector<std::string_view>(row->keys.begin(),
                                                              row->keys.end());
    };

    EXPECT_TRUE(keys_of("yolo_txt").empty());
    EXPECT_TRUE(keys_of("cifar").empty());
    EXPECT_EQ(keys_of("coco_json"),
              (std::vector<std::string_view>{"annotations", "images", "polygons"}));
    EXPECT_EQ(keys_of("voc_xml"),
              (std::vector<std::string_view>{"class_names", "keep_difficult"}));
    EXPECT_EQ(keys_of("mask_png"), (std::vector<std::string_view>{"images", "masks",
                                                                  "image_suffix",
                                                                  "mask_suffix"}));
    EXPECT_EQ(keys_of("csv"), (std::vector<std::string_view>{"file", "features", "targets"}));
}

/** @brief Tests that a row reads its keys of the request's section and names a bad one. */
TEST(Unit_DatasetsReaderTable, ARowReadsItsKeysOfTheSection)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_keys");
    scratch.write_text("rows/set.csv", "a,b,c\n1,2,3\n");

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(
        R"({"file": "rows/set.csv", "features": ["c"], "targets": ["a", "b"]})");

    const ReaderEntry* const csv = find_reader("csv");
    ASSERT_NE(csv, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, csv->read(request));
    EXPECT_EQ(list.table.values, (std::vector<float>{3.0f, 1.0f, 2.0f}));

    request.section = SaTest::attributes_from(R"({"features": "c"})");
    SA_EXPECT_ERROR((void)csv->read(request), "data 'features'", "array");

    const ReaderEntry* const coco = find_reader("coco_json");
    ASSERT_NE(coco, nullptr);
    request.section = SaTest::attributes_from(R"({"polygons": 3})");
    SA_EXPECT_ERROR((void)coco->read(request), "data 'polygons'", "boolean");
}

/** @brief Tests that the mask_png row passes the class count and the folders it is given. */
TEST(Unit_DatasetsReaderTable, TheMaskPNGRowPassesTheClassCountAndTheFolders)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_masks");
    scratch.write("pictures/train/a.jpg", {1});
    scratch.write("labels/train/a.png", {1});

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(R"({"images": "pictures", "masks": "labels"})");

    const ReaderEntry* const row = find_reader("mask_png");
    ASSERT_NE(row, nullptr);
    SA_EXPECT_FAILURE(row->read(request), StatusCode::InvalidDescription, "read_mask_png",
                      "the class count is 0");

    request.class_count = 5;
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, row->read(request));
    EXPECT_EQ(list.class_count, 5);
    EXPECT_EQ(list.samples.size(), 1u);
}

/** @brief Tests that the voc_xml row passes the class list and the difficult flag. */
TEST(Unit_DatasetsReaderTable, TheVOCXMLRowPassesTheClassListAndTheFlag)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_table_voc");
    scratch.write_text("ImageSets/Main/train.txt", "a\n");
    scratch.write_text(
        "Annotations/a.xml",
        "<annotation><size><width>8</width><height>8</height></size><object><name>dog</name>"
        "<difficult>1</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>4</xmax>"
        "<ymax>4</ymax></bndbox></object></annotation>");

    ReaderRequest request;
    request.root = scratch.path();
    request.section = SaTest::attributes_from(R"({"class_names": ["cat", "dog"]})");

    const ReaderEntry* const row = find_reader("voc_xml");
    ASSERT_NE(row, nullptr);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList dropped, row->read(request));
    EXPECT_EQ(dropped.class_count, 2);
    EXPECT_TRUE(dropped.samples[0].objects.is_empty());

    request.section = SaTest::attributes_from(
        R"({"class_names": ["cat", "dog"], "keep_difficult": true})");
    SA_ASSERT_OK_AND_ASSIGN(const SampleList kept, row->read(request));
    ASSERT_EQ(kept.samples[0].objects.size(), 1u);
    EXPECT_EQ(kept.samples[0].objects.classes()[0], 1);
}

/** @brief Tests that the cifar row refuses a class count no CIFAR labels have. */
TEST(Unit_DatasetsReaderTable, TheCIFARRowRefusesAnotherClassCount)
{
    ReaderRequest request;
    request.root = "sushiai_absent_cifar";
    request.class_count = 7;

    const ReaderEntry* const cifar = find_reader("cifar");
    ASSERT_NE(cifar, nullptr);
    SA_EXPECT_FAILURE(cifar->read(request), StatusCode::InvalidDescription, "7 classes",
                      "CIFAR-10", "coarse");
}

/** @brief Tests that detection letterboxes and reads max_objects, and classification resizes. */
TEST(Unit_DatasetsTaskTable, EachTaskStatesItsFitAndItsKeys)
{
    const TaskEntry* const detection = find_task(K_TASK_DETECTION);
    ASSERT_NE(detection, nullptr);
    EXPECT_EQ(detection->fit, SampleFit::Letterbox);
    ASSERT_EQ(detection->keys.size(), 1u);
    EXPECT_EQ(detection->keys[0], "max_objects");

    const TaskEntry* const classification = find_task(K_TASK_CLASSIFICATION);
    ASSERT_NE(classification, nullptr);
    EXPECT_EQ(classification->fit, SampleFit::Resize);
    EXPECT_TRUE(classification->keys.empty());

    const TaskEntry* const binary = find_task(K_TASK_BINARY);
    ASSERT_NE(binary, nullptr);
    EXPECT_EQ(binary->fit, SampleFit::Resize);
    ASSERT_EQ(binary->keys.size(), 1u);
    EXPECT_EQ(binary->keys[0], "positive");

    const TaskEntry* const segmentation = find_task(K_TASK_SEGMENTATION);
    ASSERT_NE(segmentation, nullptr);
    EXPECT_EQ(segmentation->fit, SampleFit::Letterbox);
    ASSERT_EQ(segmentation->keys.size(), 2u);
    EXPECT_EQ(segmentation->keys[0], "ignore_index");
    EXPECT_EQ(segmentation->keys[1], "background");

    const std::vector<std::string_view> expected = {"detection", "classification", "binary",
                                                    "multi_label", "regression", "segmentation"};
    EXPECT_EQ(names_of(task_table()), expected);
    EXPECT_EQ(find_task("pose"), nullptr);
}

/** @brief Tests that a task row builds the collator of its task from one request. */
TEST(Unit_DatasetsTaskTable, ARowBuildsTheCollatorOfItsTask)
{
    const Config::Value section = SaTest::attributes_from(R"({"max_objects": 5})");

    // One list serves every row: two named classes, and a table of two features and a target.
    SampleList list;
    list.class_names = {"cat", "dog"};
    list.class_count = 2;
    list.table.feature_count = 2;
    list.table.target_count = 1;
    const CollatorRequest request{list, 2, ImageSize{4, 4}};

    for (const TaskEntry& row : task_table())
    {
        EXPECT_NE(row.build, nullptr) << row.name;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<const Collator> collator,
                                row.make_collator(section, "data", request));
        EXPECT_EQ(collator->task(), row.name);
    }
}

/** @brief Tests that the binary row reads "positive" as a class name or as an index. */
TEST(Unit_DatasetsTaskTable, TheBinaryRowReadsThePositiveClass)
{
    SampleList list;
    list.class_names = {"cat", "dog"};
    list.class_count = 2;

    const TaskEntry* const binary = find_task(K_TASK_BINARY);
    ASSERT_NE(binary, nullptr);

    for (const char* text : {R"({})", R"({"positive": "cat"})", R"({"positive": 0})"})
    {
        const Config::Value section = SaTest::attributes_from(text);
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<const Collator> collator,
            binary->make_collator(section, "data", CollatorRequest{list, 2, ImageSize{}}));
        EXPECT_EQ(collator->task(), K_TASK_BINARY) << text;
    }

    SA_EXPECT_FAILURE(binary->make_collator(SaTest::attributes_from(R"({"positive": "bird"})"),
                                            "data", CollatorRequest{list, 2, ImageSize{}}),
                      StatusCode::InvalidDescription, "data 'positive' is 'bird'", "'cat', 'dog'");
    SA_EXPECT_FAILURE(binary->make_collator(SaTest::attributes_from(R"({"positive": 2})"),
                                            "data", CollatorRequest{list, 2, ImageSize{}}),
                      StatusCode::InvalidDescription, "the positive class is 2");
    SA_EXPECT_FAILURE(binary->make_collator(SaTest::attributes_from(R"({})"), "data",
                                            CollatorRequest{list, 3, ImageSize{}}),
                      StatusCode::InvalidDescription, "data 'task' is 'binary'",
                      "the samples have 3");
}

/** @brief Tests that the segmentation row reads "ignore_index" and refuses one that is a class. */
TEST(Unit_DatasetsTaskTable, TheSegmentationRowReadsTheIgnoreIndex)
{
    const SampleList list;
    const CollatorRequest request{list, 3, ImageSize{4, 4}};

    const TaskEntry* const segmentation = find_task(K_TASK_SEGMENTATION);
    ASSERT_NE(segmentation, nullptr);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<const Collator> collator,
        segmentation->make_collator(SaTest::attributes_from(R"({"ignore_index": -1})"), "data",
                                    request));
    EXPECT_EQ(collator->targets()[0].sample_shape, (Shape{4, 4}));

    SA_EXPECT_FAILURE(segmentation->make_collator(
                          SaTest::attributes_from(R"({"ignore_index": 1})"), "data", request),
                      StatusCode::InvalidDescription, "the ignore index 1 is one of the 3");
    SA_EXPECT_FAILURE(
        segmentation->make_collator(SaTest::attributes_from(R"({"ignore_index": 4294967296})"),
                                    "data", request),
        StatusCode::InvalidDescription, "data 'ignore_index' is 4294967296", "32 bits");
}

/** @brief Tests that a detection section over a YOLO split yields letterboxed batches. */
TEST_F(Unit_DatasetsReaderSection, ADetectionSectionBuildsLetterboxedBatches)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_detection");
    write_yolo_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "split": "train", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3)")));

    EXPECT_EQ(dataset->name(), "yolo_txt");
    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), (Shape{4, 4, 3}));

    const Data::TargetDescription* const boxes =
        Data::find_target(dataset->targets(), Data::K_ROLE_BOXES);
    const Data::TargetDescription* const classes =
        Data::find_target(dataset->targets(), Data::K_ROLE_CLASSES);
    ASSERT_NE(boxes, nullptr);
    ASSERT_NE(classes, nullptr);
    EXPECT_EQ(boxes->sample_shape, (Shape{2, 4}));
    EXPECT_EQ(classes->classes, 3);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // Row 0 of each 4 by 4 by 3 input is padding of 114, which only a letterbox leaves.
    constexpr float K_SCALE_TOLERANCE = 1e-6f;
    EXPECT_NEAR(batch.inputs.data_as<float>()[0], 114.0f / 255.0f, K_SCALE_TOLERANCE);

    const Data::BatchTarget* const valid = Data::find_target(batch.targets, Data::K_ROLE_VALID);
    const Data::BatchTarget* const read = Data::find_target(batch.targets, Data::K_ROLE_CLASSES);
    ASSERT_NE(valid, nullptr);
    ASSERT_NE(read, nullptr);
    EXPECT_EQ(valid->values.data_as<float>()[0], 0.0f);
    EXPECT_EQ(valid->values.data_as<float>()[2], 1.0f);
    EXPECT_EQ(read->values.data_as<std::int32_t>()[2], 2);
}

/** @brief Tests that a classification section takes the class count of its folders. */
TEST_F(Unit_DatasetsReaderSection, AClassificationSectionTakesTheClassesOfTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_folder");
    write_folder_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 3], "channels": 1)")));

    EXPECT_EQ(dataset->sample_shape(), (Shape{2, 3, 1}));
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_LABELS);
    EXPECT_EQ(dataset->targets()[0].classes, 2);
}

/** @brief Tests that a section may state more classes than its samples name, and not fewer. */
TEST_F(Unit_DatasetsReaderSection, TheClassCountMayExceedTheListAndNotFallBelowIt)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_classes");
    write_folder_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 2], "classes": 5)")));
    EXPECT_EQ(dataset->targets()[0].classes, 5);

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "batch_size": 2, "input": [2, 2], "classes": 1)")),
        StatusCode::InvalidDescription, "data 'classes' is 1", "name 2 classes");
}

/** @brief Tests that a reader the table lacks is refused with every reader it holds. */
TEST_F(Unit_DatasetsReaderSection, AnUnknownReaderListsTheKnownOnes)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_reader");

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "tfrecord", "task": "detection",
            "input": [4, 4], "max_objects": 2)")),
        StatusCode::InvalidDescription, "data 'reader'", "'tfrecord'", "'yolo_txt'",
        "'image_folder'", "'idx'", "'cifar'", "'coco_json'", "'voc_xml'", "'mask_png'", "'csv'");
}

/** @brief Tests that a task the table lacks is refused with every task it holds. */
TEST_F(Unit_DatasetsReaderSection, AnUnknownTaskListsTheKnownOnes)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_task");

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "pose",
            "input": [4, 4])")),
        StatusCode::InvalidDescription, "data 'task'", "'pose'", "'detection'",
        "'classification'", "'binary'", "'multi_label'", "'regression'", "'segmentation'");
}

/** @brief Tests that a key no section reads, and a key of another task, are both refused. */
TEST_F(Unit_DatasetsReaderSection, AKeyTheTaskDoesNotReadIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_keys");

    SA_EXPECT_ERROR(
        (void)build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "input": [4, 4], "max_objects": 2, "batch_sizes": 4)")),
        "data", "'batch_sizes'", "'batch_size'");
    SA_EXPECT_ERROR(
        (void)build(section_of(scratch, R"(, "reader": "image_folder",
            "task": "classification", "input": [4, 4], "max_objects": 2)")),
        "data", "'max_objects'");
}

/** @brief Tests that a detection section without its object slots is refused by key. */
TEST_F(Unit_DatasetsReaderSection, ADetectionSectionNeedsMaxObjects)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_slots");
    write_yolo_split(scratch);

    SA_EXPECT_ERROR(
        (void)build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 2, "input": [4, 4], "classes": 3)")),
        "data", "'max_objects'");
}

/** @brief Tests that an input of another length and a count below its bound are refused. */
TEST_F(Unit_DatasetsReaderSection, ValuesOutsideTheirBoundsAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_bounds");

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "input": [4, 4, 3])")),
        StatusCode::InvalidDescription, "data 'input'", "3 entries", "[height, width]");
    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "input": [4, 0])")),
        StatusCode::InvalidDescription, "data 'input[1]' is 0");
    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "classification",
            "input": [4, 4], "channels": 0)")),
        StatusCode::InvalidDescription, "data 'channels' is 0");
}

/** @brief Tests that the failure of the reader reaches the caller with its path. */
TEST_F(Unit_DatasetsReaderSection, TheFailureOfTheReaderIsReturned)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_absent");

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "input": [4, 4], "max_objects": 2, "classes": 3)")),
        StatusCode::FileNotFound, "train.txt", "images/train");
}

/** @brief Tests that a regression section over a CSV file yields rows and their targets. */
TEST_F(Unit_DatasetsReaderSection, ARegressionSectionReadsTheRowsOfACSVFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_regression");
    scratch.write_text("train.csv", "a,b,y\n1,2,3\n4,5,6\n7,8,9\n");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "csv", "task": "regression",
            "batch_size": 2, "targets": ["y"])")));

    EXPECT_EQ(dataset->name(), "csv");
    EXPECT_EQ(dataset->batch_count(), 1u);
    EXPECT_EQ(dataset->sample_shape(), Shape{2});
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_VALUES);
    EXPECT_EQ(dataset->targets()[0].sample_shape, Shape{1});

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The cells are whole numbers copied into the batch, so they compare exactly.
    const float* inputs = batch.inputs.data_as<float>();
    const float* values = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(std::vector<float>(inputs, inputs + 4),
              (std::vector<float>{1.0f, 2.0f, 4.0f, 5.0f}));
    EXPECT_EQ(std::vector<float>(values, values + 2), (std::vector<float>{3.0f, 6.0f}));
}

/** @brief Tests that a segmentation section letterboxes a mask and pads with the ignore index. */
TEST_F(Unit_DatasetsReaderSection, ASegmentationSectionFitsTheMaskOfEachImage)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_segmentation");
    Support::write_raw_image(scratch, "images/train/a.saim",
                             Support::uniform_image(ImageSize{2, 4}, 3, 255));
    Support::write_raw_image(scratch, "masks/train/a.png",
                             Support::gray_image(ImageSize{2, 4}, {0, 1, 2, 0, 1, 1, 2, 2}));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "mask_png", "task": "segmentation",
            "batch_size": 1, "input": [4, 4], "classes": 3, "ignore_index": -1)")));

    EXPECT_EQ(dataset->sample_shape(), (Shape{4, 4, 3}));
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_MASK);
    EXPECT_EQ(dataset->targets()[0].sample_shape, (Shape{4, 4}));
    EXPECT_EQ(dataset->targets()[0].classes, 3);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The 2 by 4 mask sits in rows 1 and 2, as its image does; the padding is the ignore index.
    const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 16),
              (std::vector<std::int32_t>{-1, -1, -1, -1, 0, 1, 2, 0, 1, 1, 2, 2, -1, -1, -1, -1}));
}

/** @brief Tests that a mask of another size than its image is refused with the mask file. */
TEST_F(Unit_DatasetsReaderSection, AMaskOfAnotherSizeThanItsImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_mask_size");
    Support::write_raw_image(scratch, "images/train/a.saim",
                             Support::uniform_image(ImageSize{2, 4}, 3, 255));
    Support::write_raw_image(scratch, "masks/train/a.png",
                             Support::uniform_image(ImageSize{2, 2}, 1, 0));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "mask_png", "task": "segmentation",
            "batch_size": 1, "input": [4, 4], "classes": 3)")));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::Incompatible, "masks/train/a.png",
                      "fit_mask", "the mask is 2 by 2 pixels; its image is 2 by 4");
}

/** @brief Tests that a binary section writes 1 for the class its "positive" names. */
TEST_F(Unit_DatasetsReaderSection, ABinarySectionWritesOneForThePositiveFolder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_binary");
    Support::write_raw_image(scratch, "train/cat/a.saim",
                             Support::uniform_image(ImageSize{2, 2}, 1, 10));
    Support::write_raw_image(scratch, "train/dog/b.saim",
                             Support::uniform_image(ImageSize{2, 2}, 1, 20));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "binary",
            "batch_size": 2, "input": [2, 2], "channels": 1, "positive": "cat")")));

    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].dtype, DType::Float32);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    const float* labels = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(labels[0], 1.0f);
    EXPECT_EQ(labels[1], 0.0f);
}

/** @brief Tests that a multi-label section flags the classes of each image's objects. */
TEST_F(Unit_DatasetsReaderSection, AMultiLabelSectionFlagsTheClassesOfTheObjects)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_multi_label");
    write_yolo_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "multi_label",
            "batch_size": 2, "input": [4, 4], "classes": 3)")));

    EXPECT_EQ(dataset->targets()[0].sample_shape, Shape{3});

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    const float* labels = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(std::vector<float>(labels, labels + 6),
              (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
}

/** @brief Tests that a task of images over rows of numbers, and the reverse, are refused. */
TEST_F(Unit_DatasetsReaderSection, ATaskAndAReaderOfAnotherSampleKindAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_kinds");
    scratch.write_text("train.csv", "a,y\n1,2\n");
    write_folder_split(scratch);

    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "csv", "task": "classification",
            "input": [2, 2])")),
        StatusCode::InvalidDescription, "data 'task' is 'classification'", "reads images",
        "the reader 'csv' lists rows of numbers");
    SA_EXPECT_FAILURE(
        build(section_of(scratch, R"(, "reader": "image_folder", "task": "regression")")),
        StatusCode::InvalidDescription, "data 'task' is 'regression'", "reads rows of numbers",
        "the reader 'image_folder' lists none");
}

/** @brief Tests that a key of another reader is refused and the keys allowed are listed. */
TEST_F(Unit_DatasetsReaderSection, AKeyOfAnotherReaderIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_reader_keys");

    SA_EXPECT_ERROR(
        (void)build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "input": [4, 4], "max_objects": 2, "features": ["a"])")),
        "data", "'features'", "'max_objects'");
}

/** @brief Tests that "cache" writes raw image files and that the batches are read from them. */
TEST_F(Unit_DatasetsReaderSection, ACacheIsWrittenAndRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_cache");
    write_yolo_split(scratch);
    const std::filesystem::path cache = scratch.path() / "cache";

    const std::string members = R"(, "reader": "yolo_txt", "task": "detection",
        "batch_size": 2, "input": [4, 4], "max_objects": 2, "classes": 3, "cache": ")" +
                                cache.generic_string() + "\"";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            build(section_of(scratch, members)));

    std::size_t cached = 0;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(cache))
        cached += entry.path().extension() == ".saim" ? 1 : 0;
    EXPECT_EQ(cached, 2u);

    // The sources are gone, so the batch can only come from the cache.
    std::filesystem::remove_all(scratch.path() / "images");

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    constexpr float K_SCALE_TOLERANCE = 1e-6f;
    EXPECT_NEAR(batch.inputs.data_as<float>()[12], 1.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that "shuffle", "seed" and "pipeline" build a dataset that reads a batch. */
TEST_F(Unit_DatasetsReaderSection, AShuffledSectionWithThePipelineReadsABatch)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_pipeline");
    write_yolo_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "detection",
            "batch_size": 2, "input": [16, 16], "max_objects": 8, "classes": 3,
            "shuffle": true, "seed": 7, "pipeline": "default")")));

    Data::begin_epoch(*dataset, 1);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    EXPECT_EQ(dataset->sample_shape(), (Shape{16, 16, 3}));
    EXPECT_EQ(batch.inputs.num_elements, 2 * 16 * 16 * 3);
}

/** @brief Tests that an unknown pipeline lists the rows, and that default needs a square input. */
TEST_F(Unit_DatasetsReaderSection, APipelineTheTableLacksIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_pipeline_names");
    write_yolo_split(scratch);

    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "yolo_txt",
            "task": "detection", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3, "pipeline": "mixup")")),
                      StatusCode::InvalidDescription, "data 'pipeline' is 'mixup'", "'none'",
                      "'default'");
    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "yolo_txt",
            "task": "detection", "batch_size": 2, "input": [4, 6], "max_objects": 2,
            "classes": 3, "pipeline": "default")")),
                      StatusCode::InvalidDescription, "square", "4 by 6");
}

/** @brief Tests that "workers" puts the dataset behind a loader that reads the same batches. */
TEST_F(Unit_DatasetsReaderSection, WorkersReadTheSameBatchesAhead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_workers");
    write_yolo_split(scratch);

    const std::string_view members = R"(, "reader": "yolo_txt", "task": "detection",
        "batch_size": 1, "input": [4, 4], "max_objects": 2, "classes": 3)";

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> direct,
                            build(section_of(scratch, members)));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> threaded,
        build(section_of(scratch, std::string(members) + R"(, "workers": 2, "prefetch": 2)")));

    EXPECT_NE(dynamic_cast<const Data::ThreadedLoader*>(threaded.get()), nullptr);
    EXPECT_EQ(dynamic_cast<const Data::ThreadedLoader*>(direct.get()), nullptr);
    ASSERT_EQ(threaded->batch_count(), direct->batch_count());

    Data::Batch expected;
    Data::Batch read;
    for (std::size_t index = 0; index < direct->batch_count(); ++index)
    {
        ASSERT_TRUE(SaTest::status_is_ok(direct->read(index, expected)));
        ASSERT_TRUE(SaTest::status_is_ok(threaded->read(index, read)));
        ASSERT_EQ(read.inputs.num_elements, expected.inputs.num_elements);

        const float* const a = expected.inputs.data_as<float>();
        const float* const b = read.inputs.data_as<float>();
        for (std::int64_t i = 0; i < expected.inputs.num_elements; ++i)
            ASSERT_EQ(a[i], b[i]) << "batch " << index << ", element " << i;
    }
}

/** @brief Tests the refusals of the loader keys and of an image key on a section of rows. */
TEST_F(Unit_DatasetsReaderSection, KeysThatDoNotApplyAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_misplaced");
    write_yolo_split(scratch);
    scratch.write_text("rows/set.csv", "a,b\n1,2\n3,4\n");

    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "yolo_txt",
            "task": "detection", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3, "prefetch": 4)")),
                      StatusCode::InvalidDescription, "data 'prefetch'", "data 'workers' is 0");
    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "yolo_txt",
            "task": "detection", "batch_size": 2, "input": [4, 4], "max_objects": 2,
            "classes": 3, "cache_side": 320)")),
                      StatusCode::InvalidDescription, "data 'cache_side'", "data 'cache'");
    SA_EXPECT_FAILURE(build(section_of(scratch, R"(, "reader": "csv", "task": "regression",
            "batch_size": 1, "file": "rows/set.csv", "features": ["a"], "targets": ["b"],
            "shuffle": true)")),
                      StatusCode::InvalidDescription, "data 'shuffle'", "'regression'",
                      "rows of numbers");
}

/** @brief Tests that a segmentation section over polygons paints them into the mask. */
TEST_F(Unit_DatasetsReaderSection, ASegmentationSectionPaintsThePolygonsOfALabelFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_section_polygons");
    Support::write_raw_image(scratch, "images/train/a.saim",
                             Support::uniform_image(ImageSize{4, 4}, 3, 255));
    scratch.write_text("labels/train/a.txt", "1 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5\n");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        build(section_of(scratch, R"(, "reader": "yolo_txt", "task": "segmentation",
            "batch_size": 1, "input": [4, 4], "classes": 3, "ignore_index": -1,
            "background": 2)")));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // The polygon is the top left quarter; every other pixel takes the background.
    const std::int32_t* mask = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(std::vector<std::int32_t>(mask, mask + 16),
              (std::vector<std::int32_t>{1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}));
}
