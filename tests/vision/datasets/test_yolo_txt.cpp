/**************************************************************************/
/* test_yolo_txt.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/


/**
 * @file test_yolo_txt.cpp
 * @brief Tests the reader of a dataset kept as images and one label text file per image.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/yolo_txt.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::NormalizedBox;

namespace
{
    /**
     * @brief Holds the absolute difference a parsed coordinate may have from its text: 32 times the
     *        float unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_COORDINATE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /** @brief Returns options that read one split of the folder data with three classes. */
    YOLOTxtOptions options_of(const char* split)
    {
        YOLOTxtOptions options;
        options.root = "data";
        options.split = split;
        options.class_count = 3;
        return options;
    }

    /** @brief Returns options that read one split of a directory on disk with three classes. */
    YOLOTxtOptions disk_options_of(const Support::ScratchDirectory& scratch, const char* split)
    {
        YOLOTxtOptions options;
        options.root = scratch.path();
        options.split = split;
        options.class_count = 3;
        return options;
    }

    /** @brief Adds a text file below the folder data of a source. */
    void put(MemorySource& source, const std::string& relative, const std::string& text)
    {
        source.add_file("data/" + relative, text);
    }

    /** @brief Returns a source holding a one-byte file at each path below the folder data. */
    MemorySource images_of(const std::vector<std::string>& files)
    {
        MemorySource source;
        for (const std::string& file : files)
            put(source, file, "x");

        return source;
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

    /** @brief Returns a source of three images, two of them labelled, and a note. */
    MemorySource labelled_folder()
    {
        MemorySource source = images_of({"images/train/b.jpg", "images/train/a.png",
                                         "images/train/c.jpg", "images/train/notes.md"});
        put(source, "labels/train/a.txt", "2 0.5 0.25 0.2 0.1\n0 0.1 0.1 0.05 0.05\n");
        put(source, "labels/train/b.txt", "1 0.1 0.1 0.9 0.1 0.5 0.9\n");
        return source;
    }
} // namespace

/** @brief Tests that the last "images" folder becomes "labels" and the extension .txt. */
TEST(Small_DatasetsYoloTxt, TheLabelPathSwapsTheLastImagesFolder)
{
    EXPECT_EQ(label_of("data/images/train/a.jpg"), "data/labels/train/a.txt");
    EXPECT_EQ(label_of("images/x/images/a.png"), "images/x/labels/a.txt");
    EXPECT_EQ(label_of("images/a.b.jpeg"), "labels/a.b.txt");
}

/** @brief Tests that a path with no "images" folder keeps its folder. */
TEST(Small_DatasetsYoloTxt, TheLabelPathKeepsAFolderThatIsNotImages)
{
    EXPECT_EQ(label_of("data/train/a.jpg"), "data/train/a.txt");
    EXPECT_EQ(label_of("data/images"), "data/images.txt");
    EXPECT_EQ(label_of("a.jpg"), "a.txt");
}

/** @brief Tests that the image folder is listed sorted and the files that are no image skipped. */
TEST(Small_DatasetsYoloTxt, TheImageFolderIsListedSorted)
{
    const MemorySource source = labelled_folder();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    ASSERT_EQ(list.samples.size(), 3u);
    EXPECT_EQ(relative_file(list, 0, "data"), "images/train/a.png");
    EXPECT_EQ(relative_file(list, 1, "data"), "images/train/b.jpg");
    EXPECT_EQ(relative_file(list, 2, "data"), "images/train/c.jpg");
}

/** @brief Tests that a list of boxes names its class count and no class name. */
TEST(Small_DatasetsYoloTxt, TheListHoldsAClassCountAndNoClassName)
{
    const MemorySource source = labelled_folder();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    EXPECT_EQ(list.class_count, 3);
    EXPECT_TRUE(list.class_names.empty());
    EXPECT_FALSE(list.samples[0].stored.has_value());
    EXPECT_FALSE(list.samples[0].class_index.has_value());
}

/** @brief Tests that a label file of boxes gives one object per line, with its class and box. */
TEST(Small_DatasetsYoloTxt, ALabelFileOfBoxesGivesOneObjectPerLine)
{
    const MemorySource source = labelled_folder();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    const Sample& boxes = list.samples[0];
    ASSERT_EQ(boxes.objects.size(), 2u);
    EXPECT_EQ(boxes.objects.classes()[0], 2);
    EXPECT_EQ(boxes.objects.classes()[1], 0);
    const NormalizedBox& box = boxes.objects.boxes()[0];
    EXPECT_NEAR(box.x_center, 0.5f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.y_center, 0.25f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.width, 0.2f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(box.height, 0.1f, K_COORDINATE_TOLERANCE);
    EXPECT_TRUE(boxes.objects.polygons()[0].empty());
}

/** @brief Tests that a label line of more numbers gives a polygon. */
TEST(Small_DatasetsYoloTxt, ALabelLineOfPointsGivesAPolygon)
{
    const MemorySource source = labelled_folder();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    const Sample& polygon = list.samples[1];
    ASSERT_EQ(polygon.objects.size(), 1u);
    EXPECT_EQ(polygon.objects.classes()[0], 1);
    ASSERT_EQ(polygon.objects.polygons()[0].size(), 3u);
    EXPECT_NEAR(polygon.objects.polygons()[0][1].x, 0.9f, K_COORDINATE_TOLERANCE);
    EXPECT_NEAR(polygon.objects.polygons()[0][1].y, 0.1f, K_COORDINATE_TOLERANCE);
}

/** @brief Tests that an image without a label file, or with an empty one, has no object. */
TEST(Small_DatasetsYoloTxt, AnImageWithoutALabelFileHasNoObject)
{
    MemorySource source = images_of({"images/val/a.jpg", "images/val/b.jpg"});
    put(source, "labels/val/b.txt", "");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("val"), source));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    EXPECT_TRUE(list.samples[1].objects.is_empty());
}

/** @brief Tests that an image list is read in its own order, relative paths from the root. */
TEST(Small_DatasetsYoloTxt, AnImageListIsReadInItsOwnOrder)
{
    MemorySource source = images_of({"images/all/z.jpg", "images/all/a.jpg",
                                     "images/all/unlisted.jpg"});
    put(source, "labels/all/a.txt", "1 0.5 0.5 0.5 0.5\n");
    put(source, "train.txt", "./images/all/z.jpg\r\n\n  images/all/a.jpg  \n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, "data"), "images/all/z.jpg");
    EXPECT_EQ(relative_file(list, 1, "data"), "images/all/a.jpg");
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    ASSERT_EQ(list.samples[1].objects.size(), 1u);
    EXPECT_EQ(list.samples[1].objects.classes()[0], 1);
}

/** @brief Tests that an image list wins over an image folder of the same split. */
TEST(Small_DatasetsYoloTxt, AnImageListWinsOverAnImageFolder)
{
    MemorySource source = images_of({"images/train/a.jpg", "images/train/b.jpg"});
    put(source, "train.txt", "images/train/b.jpg\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_yolo_txt(options_of("train"), source));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(relative_file(list, 0, "data"), "images/train/b.jpg");
}

/** @brief Tests that a listed image that is no file names the list and its line. */
TEST(Small_DatasetsYoloTxt, AListedImageThatIsAbsentNamesItsLine)
{
    MemorySource source = images_of({"images/a.jpg"});
    put(source, "train.txt", "images/a.jpg\n\nimages/gone.jpg\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of("train"), source), StatusCode::FileNotFound,
                      "train.txt", "line 3", "gone.jpg");
}

/** @brief Tests that a label file the parser rejects names its path and its line. */
TEST(Small_DatasetsYoloTxt, ABadLabelFileNamesItsPathAndLine)
{
    MemorySource source = images_of({"images/train/a.jpg", "images/train/b.jpg"});
    put(source, "labels/train/a.txt", "0 0.5 0.5 0.2 0.4\n");
    put(source, "labels/train/b.txt", "0 0.5 0.5 0.2 0.4\n1 0.5 abc 0.2 0.4\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of("train"), source), StatusCode::InvalidFile,
                      "labels/train/b.txt", "line 2", "abc");
}

/** @brief Tests that a class at or above the class count is refused with its label file. */
TEST(Small_DatasetsYoloTxt, AClassOutsideTheClassCountIsRefused)
{
    MemorySource source = images_of({"images/train/a.jpg"});
    put(source, "labels/train/a.txt", "3 0.5 0.5 0.2 0.4\n");

    SA_EXPECT_FAILURE(read_yolo_txt(options_of("train"), source), StatusCode::InvalidFile,
                      "labels/train/a.txt", "line 1");
}

/** @brief Tests that a class count below one is refused before any file is read. */
TEST(Small_DatasetsYoloTxt, AClassCountBelowOneIsRefused)
{
    const MemorySource source;
    YOLOTxtOptions options = options_of("train");
    options.class_count = 0;

    SA_EXPECT_FAILURE(read_yolo_txt(options, source), StatusCode::InvalidDescription,
                      "class count is 0");
}

/** @brief Tests that a split with neither a list nor a folder names both paths. */
TEST(Small_DatasetsYoloTxt, ASplitWithNeitherFormIsNotFound)
{
    const MemorySource source;

    SA_EXPECT_FAILURE(read_yolo_txt(options_of("train"), source), StatusCode::FileNotFound,
                      "train.txt", "images/train");
}

/** @brief Tests that an image folder with no image file is refused. */
TEST(Small_DatasetsYoloTxt, ASplitWithoutAnImageIsRefused)
{
    const MemorySource source = images_of({"images/train/readme.txt"});

    SA_EXPECT_FAILURE(read_yolo_txt(options_of("train"), source), StatusCode::InvalidFile,
                      "'train'", "no image");
}

/** @brief Tests that the file-system form lists the images and pairs the labels on disk. */
TEST(Medium_DatasetsYoloTxt, ImagesAndLabelsOnDiskArePaired)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_disk");
    scratch.write("images/train/b.jpg", {1});
    scratch.write("images/train/a.png", {1});
    scratch.write_text("labels/train/b.txt", "2 0.5 0.25 0.2 0.1\n");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list,
                            read_yolo_txt(disk_options_of(scratch, "train")));

    ASSERT_EQ(list.samples.size(), 2u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "images/train/a.png");
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    ASSERT_EQ(list.samples[1].objects.size(), 1u);
    EXPECT_EQ(list.samples[1].objects.classes()[0], 2);
}

/** @brief Tests that a listed image that is absent from the disk names the list and its line. */
TEST(Medium_DatasetsYoloTxt, AListedImageAbsentFromTheDiskNamesItsLine)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_list_absent");
    scratch.write("images/a.jpg", {1});
    scratch.write_text("train.txt", "images/a.jpg\n\nimages/gone.jpg\n");

    SA_EXPECT_FAILURE(read_yolo_txt(disk_options_of(scratch, "train")), StatusCode::FileNotFound,
                      "train.txt", "line 3", "gone.jpg");
}

/** @brief Tests that a split with neither a list nor a folder on disk names both paths. */
TEST(Medium_DatasetsYoloTxt, ASplitWithNeitherFormOnDiskIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_yolo_absent");

    SA_EXPECT_FAILURE(read_yolo_txt(disk_options_of(scratch, "train")), StatusCode::FileNotFound,
                      "train.txt", "images/train");
}
