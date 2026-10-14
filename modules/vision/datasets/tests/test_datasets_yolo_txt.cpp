/**************************************************************************/
/* test_datasets_yolo_txt.cpp                                             */
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
 * @file test_datasets_yolo_txt.cpp
 * @brief Tests the reader of a dataset kept as images and one label text file per image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/yolo_txt.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /** @brief Holds the absolute difference a parsed coordinate may have from its text. */
    constexpr float K_COORDINATE_TOLERANCE = 1e-6f;

    /** @brief Returns options that read one split of a directory with three classes. */
    YOLOTxtOptions options_of(const Support::ScratchDirectory& scratch, const char* split)
    {
        YOLOTxtOptions options;
        options.root = scratch.path();
        options.split = split;
        options.class_count = 3;
        return options;
    }

    /** @brief Returns the label path of an image, with forward slashes. */
    std::string label_of(const char* image)
    {
        return yolo_label_path(image).generic_string();
    }

    /** @brief Returns the path of a sample's file below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that the last "images" folder becomes "labels" and the extension .txt. */
TEST(Unit_DatasetsYoloTxt, TheLabelPathSwapsTheLastImagesFolder)
{
    EXPECT_EQ(label_of("data/images/train/a.jpg"), "data/labels/train/a.txt");
    EXPECT_EQ(label_of("images/x/images/a.png"), "images/x/labels/a.txt");
    EXPECT_EQ(label_of("images/a.b.jpeg"), "labels/a.b.txt");
}

/** @brief Tests that a path with no "images" folder keeps its folder. */
TEST(Unit_DatasetsYoloTxt, TheLabelPathKeepsAFolderThatIsNotImages)
{
    EXPECT_EQ(label_of("data/train/a.jpg"), "data/train/a.txt");
    EXPECT_EQ(label_of("data/images"), "data/images.txt");
    EXPECT_EQ(label_of("a.jpg"), "a.txt");
}

/** @brief Tests that the image folder is listed sorted and each label file is parsed. */
TEST(Unit_DatasetsYoloTxt, TheImageFolderIsListedWithItsLabels)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_folder");
    scratch.write("images/train/b.jpg", {1});
    scratch.write("images/train/a.png", {1});
    scratch.write("images/train/c.jpg", {1});
    scratch.write("images/train/notes.md", {1});
    scratch.write_text("labels/train/a.txt", "2 0.5 0.25 0.2 0.1\n0 0.1 0.1 0.05 0.05\n");
    scratch.write_text("labels/train/b.txt", "1 0.1 0.1 0.9 0.1 0.5 0.9\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of(scratch, "train")));

    EXPECT_EQ(list.class_count, 3);
    EXPECT_TRUE(list.class_names.empty());
    ASSERT_EQ(list.samples.size(), 3u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "images/train/a.png");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "images/train/b.jpg");
    EXPECT_EQ(relative_file(list, 2, scratch.path()), "images/train/c.jpg");

    const Sample& boxes = list.samples[0];
    EXPECT_FALSE(boxes.stored.has_value());
    EXPECT_FALSE(boxes.class_index.has_value());
    ASSERT_EQ(boxes.objects.size(), 2u);
    EXPECT_EQ(boxes.objects.classes()[0], 2);
    EXPECT_EQ(boxes.objects.classes()[1], 0);

    const NormalizedBox& box = boxes.objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 0.5f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.y_center, 0.25f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.width, 0.2f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.height, 0.1f, K_COORDINATE_TOLERANCE);
    EXPECT_TRUE(boxes.objects.polygons()[0].empty());

    const Sample& polygon = list.samples[1];
    ASSERT_EQ(polygon.objects.size(), 1u);
    EXPECT_EQ(polygon.objects.classes()[0], 1);
    ASSERT_EQ(polygon.objects.polygons()[0].size(), 3u);
    EXPECT_NEAR(polygon.objects.polygons()[0][1].x, 0.9f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(polygon.objects.polygons()[0][1].y, 0.1f, K_COORDINATE_TOLERANCE);
}

/** @brief Tests that an image without a label file, or with an empty one, has no object. */
TEST(Unit_DatasetsYoloTxt, AnImageWithoutALabelFileHasNoObject)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_background");
    scratch.write("images/val/a.jpg", {1});
    scratch.write("images/val/b.jpg", {1});
    scratch.write_text("labels/val/b.txt", "");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of(scratch, "val")));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    EXPECT_TRUE(list.samples[1].objects.is_empty());
}

/** @brief Tests that an image list is read in its own order, relative paths from the root. */
TEST(Unit_DatasetsYoloTxt, AnImageListIsReadInItsOwnOrder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_list");
    scratch.write("images/all/z.jpg", {1});
    scratch.write("images/all/a.jpg", {1});
    scratch.write("images/all/unlisted.jpg", {1});
    scratch.write_text("labels/all/a.txt", "1 0.5 0.5 0.5 0.5\n");
    scratch.write_text("train.txt", "./images/all/z.jpg\r\n\n  images/all/a.jpg  \n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of(scratch, "train")));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "images/all/z.jpg");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "images/all/a.jpg");
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    ASSERT_EQ(list.samples[1].objects.size(), 1u);
    EXPECT_EQ(list.samples[1].objects.classes()[0], 1);
}

/** @brief Tests that an image list wins over an image folder of the same split. */
TEST(Unit_DatasetsYoloTxt, AnImageListWinsOverAnImageFolder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_both");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("images/train/b.jpg", {1});
    scratch.write_text("train.txt", "images/train/b.jpg\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of(scratch, "train")));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "images/train/b.jpg");
}

/** @brief Tests that a listed image that is no file names the list and its line. */
TEST(Unit_DatasetsYoloTxt, AListedImageThatIsAbsentNamesItsLine)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_list_absent");
    scratch.write("images/a.jpg", {1});
    scratch.write_text("train.txt", "images/a.jpg\n\nimages/gone.jpg\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of(scratch, "train")), StatusCode::FileNotFound,
                      "train.txt", "line 3", "gone.jpg");
}

/** @brief Tests that a label file the parser rejects names its path and its line. */
TEST(Unit_DatasetsYoloTxt, ABadLabelFileNamesItsPathAndLine)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_bad_label");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("images/train/b.jpg", {1});
    scratch.write_text("labels/train/a.txt", "0 0.5 0.5 0.2 0.4\n");
    scratch.write_text("labels/train/b.txt", "0 0.5 0.5 0.2 0.4\n1 0.5 abc 0.2 0.4\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of(scratch, "train")), StatusCode::InvalidFile,
                      "labels/train/b.txt", "line 2", "abc");
}

/** @brief Tests that a class at or above the class count is refused with its label file. */
TEST(Unit_DatasetsYoloTxt, AClassOutsideTheClassCountIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_class");
    scratch.write("images/train/a.jpg", {1});
    scratch.write_text("labels/train/a.txt", "3 0.5 0.5 0.2 0.4\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of(scratch, "train")), StatusCode::InvalidFile,
                      "labels/train/a.txt", "line 1");
}

/** @brief Tests that a class count below one is refused before the disk is read. */
TEST(Unit_DatasetsYoloTxt, AClassCountBelowOneIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_count");
    YOLOTxtOptions options = options_of(scratch, "train");
    options.class_count = 0;

    SA_EXPECT_FAILURE(read_yolo_txt(options), StatusCode::InvalidDescription, "class count is 0");
}

/** @brief Tests that a split with neither a list nor a folder names both paths. */
TEST(Unit_DatasetsYoloTxt, ASplitWithNeitherFormIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_absent");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of(scratch, "train")), StatusCode::FileNotFound,
                      "train.txt", "images/train");
}

/** @brief Tests that an image folder with no image file is refused. */
TEST(Unit_DatasetsYoloTxt, ASplitWithoutAnImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_empty");
    scratch.write("images/train/readme.txt", {1});

    SA_EXPECT_FAILURE(read_yolo_txt(options_of(scratch, "train")), StatusCode::InvalidFile,
                      "'train'", "no image");
}
