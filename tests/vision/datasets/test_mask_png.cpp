/**************************************************************************/
/* test_mask_png.cpp                                                      */
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
 * @file test_mask_png.cpp
 * @brief Tests the reader of a dataset kept as an image folder and a mask folder.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/mask_png.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read the split "train" of the folder data with four classes. */
    MaskPNGOptions options_of()
    {
        MaskPNGOptions options;
        options.root = "data";
        options.class_count = 4;
        return options;
    }

    /** @brief Returns options that read the split "train" of a directory with four classes. */
    MaskPNGOptions disk_options_of(const Support::ScratchDirectory& scratch)
    {
        MaskPNGOptions options;
        options.root = scratch.path();
        options.class_count = 4;
        return options;
    }

    /** @brief Returns a source holding a one-byte file at each path below the folder data. */
    MemorySource source_of(const std::vector<std::string>& files)
    {
        MemorySource source;
        for (const std::string& file : files)
            source.add_file("data/" + file, "x");

        return source;
    }

    /** @brief Returns a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t file,
                              const std::filesystem::path& base)
    {
        return list.files[file].lexically_relative(base).generic_string();
    }

    /** @brief Returns options that name the Cityscapes folders and suffixes. */
    MaskPNGOptions cityscapes_options()
    {
        MaskPNGOptions options = options_of();
        options.image_folder = "leftImg8bit";
        options.mask_folder = "gtFine";
        options.image_suffix = "_leftImg8bit";
        options.mask_suffix = "_gtFine_labelIds";
        return options;
    }
} // namespace

/** @brief Tests that each image, in sorted order, refers to the mask of its stem. */
TEST(Small_DatasetsMaskPng, EachImageRefersToTheMaskOfItsStem)
{
    const MemorySource source =
        source_of({"images/train/b.jpg", "images/train/a/c.jpeg", "images/train/notes.md",
                   "masks/train/b.png", "masks/train/a/c.png", "masks/train/index.txt"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options_of(), source));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_EQ(list.files.size(), 4u);
    ASSERT_TRUE(list.samples[0].mask.has_value());
    ASSERT_TRUE(list.samples[1].mask.has_value());
    EXPECT_EQ(relative_file(list, list.samples[0].file, "data"), "images/train/a/c.jpeg");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, "data"), "masks/train/a/c.png");
    EXPECT_EQ(relative_file(list, list.samples[1].file, "data"), "images/train/b.jpg");
    EXPECT_EQ(relative_file(list, *list.samples[1].mask, "data"), "masks/train/b.png");
}

/** @brief Tests that a list of masks names its class count and no class, label or object. */
TEST(Small_DatasetsMaskPng, TheListHoldsAClassCountAndNoLabel)
{
    const MemorySource source = source_of({"images/train/a.jpg", "masks/train/a.png"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options_of(), source));

    EXPECT_EQ(list.class_count, 4);
    EXPECT_TRUE(list.class_names.empty());
    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_FALSE(list.samples[0].class_index.has_value());
    EXPECT_TRUE(list.samples[0].objects.is_empty());
}

/** @brief Tests that the two folders may be named and the split left out. */
TEST(Small_DatasetsMaskPng, TheFoldersMayBeNamedAndTheSplitLeftOut)
{
    const MemorySource source = source_of({"leftImg8bit/a.png", "gtFine/a.png"});
    MaskPNGOptions options = options_of();
    options.split.clear();
    options.image_folder = "leftImg8bit";
    options.mask_folder = "gtFine";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options, source));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(relative_file(list, 0, "data"), "leftImg8bit/a.png");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, "data"), "gtFine/a.png");
}

/** @brief Tests that an image without a mask is refused with both paths. */
TEST(Small_DatasetsMaskPng, AnImageWithoutAMaskIsRefused)
{
    const MemorySource source =
        source_of({"images/train/a.jpg", "images/train/b.jpg", "masks/train/a.png"});

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::FileNotFound,
                      "images/train/b.jpg", "read_mask_png", "has no mask", "masks/train/b.png");
}

/** @brief Tests that a mask without an image is refused with its path. */
TEST(Small_DatasetsMaskPng, AMaskWithoutAnImageIsRefused)
{
    const MemorySource source =
        source_of({"images/train/a.jpg", "masks/train/a.png", "masks/train/z.png"});

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::InvalidFile,
                      "masks/train/z.png", "read_mask_png", "the mask has no image");
}

/** @brief Tests that two images of one stem, which would share a mask, are refused. */
TEST(Small_DatasetsMaskPng, TwoImagesOfOneStemAreRefused)
{
    const MemorySource source =
        source_of({"images/train/a.jpg", "images/train/a.png", "masks/train/a.png"});

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::InvalidFile,
                      "images/train/a.png", "shares the mask", "masks/train/a.png");
}

/** @brief Tests that a class count below 1 is refused. */
TEST(Small_DatasetsMaskPng, AClassCountBelowOneIsRefused)
{
    const MemorySource source = source_of({"images/train/a.jpg", "masks/train/a.png"});
    MaskPNGOptions options = options_of();
    options.class_count = 0;

    SA_EXPECT_FAILURE(read_mask_png(options, source), StatusCode::InvalidDescription,
                      "read_mask_png", "the class count is 0");
}

/** @brief Tests that an absent image folder is reported by its path. */
TEST(Small_DatasetsMaskPng, AnAbsentImageFolderIsReported)
{
    const MemorySource source = source_of({"masks/train/a.png"});

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::FileNotFound,
                      "images/train", "no such directory");
}

/** @brief Tests that an absent mask folder is reported by its path. */
TEST(Small_DatasetsMaskPng, AnAbsentMaskFolderIsReported)
{
    MemorySource source = source_of({});
    source.add_directory("data/images/train");

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::FileNotFound,
                      "masks/train", "no such directory");
}

/** @brief Tests that an image folder that holds no image is refused. */
TEST(Small_DatasetsMaskPng, AnImageFolderWithoutAnImageIsRefused)
{
    MemorySource source = source_of({});
    source.add_directory("data/images/train");
    source.add_directory("data/masks/train");

    SA_EXPECT_FAILURE(read_mask_png(options_of(), source), StatusCode::InvalidFile,
                      "images/train", "holds no image");
}

/** @brief Tests that the suffixes pair names that differ, as those of Cityscapes do. */
TEST(Small_DatasetsMaskPng, TheSuffixesPairNamesThatDiffer)
{
    const MemorySource source = source_of(
        {"leftImg8bit/train/aachen/aachen_000000_leftImg8bit.png",
         "leftImg8bit/train/bonn/bonn_000001_leftImg8bit.png",
         "gtFine/train/aachen/aachen_000000_gtFine_labelIds.png",
         "gtFine/train/aachen/aachen_000000_gtFine_color.png",
         "gtFine/train/bonn/bonn_000001_gtFine_labelIds.png"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(cityscapes_options(), source));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_EQ(list.files.size(), 4u);
    ASSERT_TRUE(list.samples[0].mask.has_value());
    ASSERT_TRUE(list.samples[1].mask.has_value());
    EXPECT_EQ(relative_file(list, list.samples[0].file, "data"),
              "leftImg8bit/train/aachen/aachen_000000_leftImg8bit.png");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, "data"),
              "gtFine/train/aachen/aachen_000000_gtFine_labelIds.png");
    EXPECT_EQ(relative_file(list, *list.samples[1].mask, "data"),
              "gtFine/train/bonn/bonn_000001_gtFine_labelIds.png");
}

/** @brief Tests that an image whose name lacks the image suffix is refused with its path. */
TEST(Small_DatasetsMaskPng, AnImageWithoutTheImageSuffixIsRefused)
{
    const MemorySource source =
        source_of({"images/train/a_left.png", "images/train/b.png", "masks/train/a_ids.png"});
    MaskPNGOptions options = options_of();
    options.image_suffix = "_left";
    options.mask_suffix = "_ids";

    SA_EXPECT_FAILURE(read_mask_png(options, source), StatusCode::InvalidFile,
                      "images/train/b.png", "read_mask_png", "does not end in '_left'");
}

/** @brief Tests that a mask of the mask suffix that no image is paired with is refused. */
TEST(Small_DatasetsMaskPng, AMaskOfTheSuffixWithoutAnImageIsRefused)
{
    const MemorySource source = source_of(
        {"images/train/a.jpg", "masks/train/a_ids.png", "masks/train/b_ids.png"});
    MaskPNGOptions options = options_of();
    options.mask_suffix = "_ids";

    SA_EXPECT_FAILURE(read_mask_png(options, source), StatusCode::InvalidFile,
                      "masks/train/b_ids.png", "the mask has no image");
}

/** @brief Tests that the file-system form pairs the images and masks written to disk. */
TEST(Medium_DatasetsMaskPng, ImagesAndMasksOnDiskArePairedByTheirStems)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_pairs");
    scratch.write("images/train/b.jpg", {1});
    scratch.write("images/train/a/c.jpeg", {1});
    scratch.write("masks/train/b.png", {1});
    scratch.write("masks/train/a/c.png", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(disk_options_of(scratch)));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_TRUE(list.samples[0].mask.has_value());
    ASSERT_TRUE(list.samples[1].mask.has_value());
    EXPECT_EQ(relative_file(list, list.samples[0].file, scratch.path()), "images/train/a/c.jpeg");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, scratch.path()), "masks/train/a/c.png");
    EXPECT_EQ(relative_file(list, list.samples[1].file, scratch.path()), "images/train/b.jpg");
    EXPECT_EQ(relative_file(list, *list.samples[1].mask, scratch.path()), "masks/train/b.png");
}

/** @brief Tests that an image folder that is absent from the disk is reported as not found. */
TEST(Medium_DatasetsMaskPng, AnImageFolderAbsentFromTheDiskIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_folders_absent");

    SA_EXPECT_FAILURE(read_mask_png(disk_options_of(scratch)), StatusCode::FileNotFound,
                      "images/train", "no such directory");
}
