/**************************************************************************/
/* test_coco_json.cpp                                                     */
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
 * @file test_coco_json.cpp
 * @brief Tests the reader of a dataset kept as one annotation JSON and an image folder.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/coco_json.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /** @brief Holds the absolute difference a fraction may have from its quotient. */
    constexpr float K_FRACTION_TOLERANCE = 1e-6f;

    /** @brief Holds two images of 200 by 100 pixels, the second listed first by id. */
    constexpr const char* K_IMAGES =
        R"("images": [{"id": 9, "file_name": "b.jpg", "width": 200, "height": 100},
                      {"id": 4, "file_name": "sub/a.jpg", "width": 200, "height": 100}])";

    /** @brief Holds three categories whose ids are sparse and not in order. */
    constexpr const char* K_CATEGORIES =
        R"("categories": [{"id": 90, "name": "brush"}, {"id": 1, "name": "person"},
                          {"id": 17, "name": "cat"}])";

    /** @brief Writes the annotation file of the split "train" and creates its image folder. */
    void write_split(const Support::ScratchDirectory& scratch, const std::string& annotations)
    {
        scratch.make_directory("train");
        scratch.write_text("annotations/instances_train.json",
                           std::string("{") + K_IMAGES + ", " + K_CATEGORIES +
                               ", \"annotations\": [" + annotations + "]}");
    }

    /** @brief Returns options that read the split "train" of a directory. */
    COCOJSONOptions options_of(const Support::ScratchDirectory& scratch)
    {
        COCOJSONOptions options;
        options.root = scratch.path();
        return options;
    }

    /** @brief Returns the path of a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that boxes become fractions and category ids dense indices in id order. */
TEST(Unit_DatasetsCocoJson, BoxesAndDenseClassesAreRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_boxes");
    write_split(scratch,
                R"({"id": 1, "image_id": 4, "category_id": 90, "bbox": [10, 20, 50, 20]},
                   {"id": 2, "image_id": 4, "category_id": 1, "bbox": [0, 0, 200, 100.0],
                    "iscrowd": 0},
                   {"id": 3, "image_id": 4, "category_id": 17, "bbox": [1, 1, 5, 5],
                    "iscrowd": 1},
                   {"id": 4, "image_id": 4, "category_id": 17, "bbox": [1, 1, 0, 5]})");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options_of(scratch)));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"person", "cat", "brush"}));
    EXPECT_EQ(list.class_count, 3);
    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "train/b.jpg");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "train/sub/a.jpg");
    EXPECT_TRUE(list.samples[0].objects.is_empty());

    // The crowd annotation and the box of no width are left out.
    const Sample& sample = list.samples[1];
    EXPECT_FALSE(sample.class_index.has_value());
    ASSERT_EQ(sample.objects.size(), 2u);
    EXPECT_EQ(sample.objects.classes()[0], 2);
    EXPECT_EQ(sample.objects.classes()[1], 0);
    EXPECT_TRUE(sample.objects.polygons()[0].empty());

    const NormalizedBox& box = sample.objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 35.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.y_center, 30.0f / 100.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.width, 50.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(box.height, 20.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that the file and the image folder may be named. */
TEST(Unit_DatasetsCocoJson, TheFileAndTheFolderMayBeNamed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_named");
    scratch.make_directory("pictures");
    scratch.write_text("labels.json", std::string("{") + K_IMAGES + ", " + K_CATEGORIES +
                                          ", \"annotations\": []}");

    COCOJSONOptions options = options_of(scratch);
    options.annotation_file = "labels.json";
    options.image_folder = "pictures";
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options));

    EXPECT_EQ(relative_file(list, 0, scratch.path()), "pictures/b.jpg");
}

/** @brief Tests that a polygon is kept on request and the largest part of several is taken. */
TEST(Unit_DatasetsCocoJson, PolygonsAreKeptWhenAskedFor)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_polygons");
    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 100, 50],
                    "segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]]},
                   {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
                    "iscrowd": 1, "segmentation": {"counts": [1, 2], "size": [100, 200]}})");

    COCOJSONOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList ignored, read_coco_json(options));
    EXPECT_TRUE(ignored.samples[0].objects.polygons()[0].empty());

    options.segmentation = COCOSegmentation::Polygons;
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options));

    ASSERT_EQ(list.samples[0].objects.size(), 1u);
    const auto& polygon = list.samples[0].objects.polygons()[0];
    ASSERT_EQ(polygon.size(), 4u);
    EXPECT_NEAR(polygon[2].x, 100.0f / 200.0f, K_FRACTION_TOLERANCE);
    EXPECT_NEAR(polygon[2].y, 50.0f / 100.0f, K_FRACTION_TOLERANCE);
}

/** @brief Tests that a run-length mask fails by its index only when polygons are asked for. */
TEST(Unit_DatasetsCocoJson, ARunLengthMaskIsRefusedWhenPolygonsAreAskedFor)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_rle");
    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10]},
                   {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
                    "segmentation": {"counts": "abc", "size": [100, 200]}})");

    COCOJSONOptions options = options_of(scratch);
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options));
    EXPECT_EQ(list.samples[0].objects.size(), 2u);

    options.segmentation = COCOSegmentation::Polygons;
    SA_EXPECT_FAILURE(read_coco_json(options), StatusCode::InvalidFile, "instances_train.json",
                      "read_coco_json", "annotation 1 holds a run-length mask");
}

/** @brief Tests that a polygon part of an odd or short number list fails by its index. */
TEST(Unit_DatasetsCocoJson, AMalformedPolygonIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_outline");
    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10],
                    "segmentation": [[0, 0, 2, 0, 2, 2, 5]]})");

    COCOJSONOptions options = options_of(scratch);
    options.segmentation = COCOSegmentation::Polygons;
    SA_EXPECT_FAILURE(read_coco_json(options), StatusCode::InvalidFile, "annotation 0",
                      "'segmentation' part", "at least 3 pairs");
}

/** @brief Tests that an annotation of an unknown image or category fails by its index. */
TEST(Unit_DatasetsCocoJson, AnUnknownImageOrCategoryNamesTheAnnotation)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_unknown");

    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10]},
                   {"id": 2, "image_id": 5, "category_id": 1, "bbox": [0, 0, 10, 10]})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json",
                      "annotation 1 has the 'image_id' 5, which 'images' lacks");

    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 2, "bbox": [0, 0, 10, 10],
                    "iscrowd": 1})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "annotation 0 has the 'category_id' 2, which 'categories' lacks");
}

/** @brief Tests that a box that is not four numbers fails by the index of its annotation. */
TEST(Unit_DatasetsCocoJson, AMalformedBoxNamesTheAnnotation)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_box");

    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10]})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "annotation 0 has no 'bbox' of 4 numbers");

    write_split(scratch,
                R"({"id": 1, "image_id": 9, "category_id": 1, "bbox": [0, 0, 10, "x"]})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "annotation 0 has no 'bbox' of 4 numbers");

    write_split(scratch, R"({"id": 1, "image_id": "9", "category_id": 1})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "annotation 0 has no whole number 'image_id'");
}

/** @brief Tests that two entries of one id, and an image of no size, fail by their indices. */
TEST(Unit_DatasetsCocoJson, TheImagesAndTheCategoriesAreChecked)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_lists");
    scratch.make_directory("train");

    scratch.write_text("annotations/instances_train.json",
                       R"({"images": [{"id": 1, "file_name": "a.jpg", "width": 2, "height": 2}],
                           "categories": [{"id": 3, "name": "a"}, {"id": 7, "name": "b"},
                                          {"id": 3, "name": "c"}],
                           "annotations": []})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "category 0 and category 2 share the id 3");

    scratch.write_text("annotations/instances_train.json",
                       R"({"images": [{"id": 1, "file_name": "a.jpg", "width": 2, "height": 2},
                                      {"id": 2, "file_name": "b.jpg", "width": 0, "height": 2}],
                           "categories": [{"id": 3, "name": "a"}], "annotations": []})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "image 1 is 2 by 0 pixels");

    scratch.write_text("annotations/instances_train.json",
                       R"({"images": [{"id": 1, "width": 2, "height": 2}],
                           "categories": [{"id": 3, "name": "a"}], "annotations": []})");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "image 0 has no string 'file_name'");
}

/** @brief Tests that an absent file, an absent folder, bad JSON and a missing list fail. */
TEST(Unit_DatasetsCocoJson, TheFileMustExistAndHoldItsThreeLists)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_file");

    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::FileNotFound, "train",
                      "read_coco_json", "no such image folder");

    scratch.make_directory("train");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::FileNotFound,
                      "annotations/instances_train.json", "read_coco_json", "no such file");

    scratch.write_text("annotations/instances_train.json", "{\"images\": [");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json", "not valid JSON");

    scratch.write_text("annotations/instances_train.json",
                       std::string("{") + K_IMAGES + ", \"annotations\": []}");
    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "the file has no list 'categories'");
}

/** @brief Tests that annotations met before their images and categories still resolve. */
TEST(Unit_DatasetsCocoJson, TheListsMayComeInAnyOrder)
{
    const std::string annotations =
        R"("annotations": [
               {"segmentation": [[0, 0, 2, 0, 2, 2], [0, 0, 100, 0, 100, 50, 0, 50]],
                "bbox": [10, 20, 50, 20], "category_id": 90, "image_id": 4, "id": 1},
               {"id": 2, "image_id": 9, "category_id": 1, "bbox": [0, 0, 200, 100]}])";

    const Support::ScratchDirectory in_order("sushiai_test_datasets_coco_in_order");
    in_order.make_directory("train");
    in_order.write_text("annotations/instances_train.json",
                        std::string("{") + K_IMAGES + ", " + K_CATEGORIES + ", " + annotations +
                            "}");

    const Support::ScratchDirectory reversed("sushiai_test_datasets_coco_reversed");
    reversed.make_directory("train");
    reversed.write_text("annotations/instances_train.json",
                        std::string("{\"info\": {\"year\": 2017, \"notes\": [1, [2]]}, ") +
                            annotations + ", " + K_CATEGORIES + ", " + K_IMAGES + "}");

    COCOJSONOptions options = options_of(in_order);
    options.segmentation = COCOSegmentation::Polygons;
    SA_ASSERT_OK_AND_ASSIGN(const SampleList expected, read_coco_json(options));

    options.root = reversed.path();
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_coco_json(options));

    EXPECT_EQ(list.class_names, expected.class_names);
    ASSERT_EQ(list.samples.size(), expected.samples.size());
    for (std::size_t sample = 0; sample < list.samples.size(); ++sample)
    {
        const auto& objects = list.samples[sample].objects;
        const auto& wanted = expected.samples[sample].objects;

        EXPECT_EQ(relative_file(list, sample, reversed.path()),
                  relative_file(expected, sample, in_order.path()));
        ASSERT_EQ(objects.size(), 1u);
        ASSERT_EQ(wanted.size(), 1u);
        EXPECT_EQ(objects.classes()[0], wanted.classes()[0]);
        EXPECT_EQ(objects.boxes()[0].x_center, wanted.boxes()[0].x_center);
        EXPECT_EQ(objects.boxes()[0].height, wanted.boxes()[0].height);
        ASSERT_EQ(objects.polygons()[0].size(), wanted.polygons()[0].size());
        for (std::size_t vertex = 0; vertex < objects.polygons()[0].size(); ++vertex)
        {
            EXPECT_EQ(objects.polygons()[0][vertex].x, wanted.polygons()[0][vertex].x);
            EXPECT_EQ(objects.polygons()[0][vertex].y, wanted.polygons()[0][vertex].y);
        }
    }

    // Image 4 is the second sample; its outline is the part of four vertices.
    EXPECT_EQ(list.samples[1].objects.classes()[0], 2);
    EXPECT_EQ(list.samples[1].objects.polygons()[0].size(), 4u);
    EXPECT_TRUE(list.samples[0].objects.polygons()[0].empty());
}

/** @brief Tests that a list the file states twice is refused by its key. */
TEST(Unit_DatasetsCocoJson, AListStatedTwiceIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_coco_twice");
    scratch.make_directory("train");
    scratch.write_text("annotations/instances_train.json",
                       std::string("{") + K_IMAGES + ", " + K_CATEGORIES +
                           ", \"annotations\": [], " + K_IMAGES + "}");

    SA_EXPECT_FAILURE(read_coco_json(options_of(scratch)), StatusCode::InvalidFile,
                      "instances_train.json", "read_coco_json", "states 'images' twice");
}
