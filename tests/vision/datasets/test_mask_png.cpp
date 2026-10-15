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
 * @file test_mask_png.cpp
 * @brief Tests the reader of a dataset kept as an image folder and a mask folder.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/mask_png.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read the split "train" of a directory with four classes. */
    MaskPNGOptions options_of(const Support::ScratchDirectory& scratch)
    {
        MaskPNGOptions options;
        options.root = scratch.path();
        options.class_count = 4;
        return options;
    }

    /** @brief Returns a file of a list below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t file,
                              const std::filesystem::path& base)
    {
        return list.files[file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that each image, in sorted order, refers to the mask of its stem. */
TEST(Small_DatasetsMaskPng, EachImageRefersToTheMaskOfItsStem)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_pairs");
    scratch.write("images/train/b.jpg", {1});
    scratch.write("images/train/a/c.jpeg", {1});
    scratch.write("images/train/notes.md", {1});
    scratch.write("masks/train/b.png", {1});
    scratch.write("masks/train/a/c.png", {1});
    scratch.write("masks/train/index.txt", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options_of(scratch)));

    EXPECT_EQ(list.class_count, 4);
    EXPECT_TRUE(list.class_names.empty());
    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_EQ(list.files.size(), 4u);

    const Sample& first = list.samples[0];
    ASSERT_TRUE(first.mask.has_value());
    EXPECT_FALSE(first.class_index.has_value());
    EXPECT_TRUE(first.objects.is_empty());
    EXPECT_EQ(relative_file(list, first.file, scratch.path()), "images/train/a/c.jpeg");
    EXPECT_EQ(relative_file(list, *first.mask, scratch.path()), "masks/train/a/c.png");

    const Sample& second = list.samples[1];
    ASSERT_TRUE(second.mask.has_value());
    EXPECT_EQ(relative_file(list, second.file, scratch.path()), "images/train/b.jpg");
    EXPECT_EQ(relative_file(list, *second.mask, scratch.path()), "masks/train/b.png");
}

/** @brief Tests that the two folders may be named and the split left out. */
TEST(Small_DatasetsMaskPng, TheFoldersMayBeNamedAndTheSplitLeftOut)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_folders");
    scratch.write("leftImg8bit/a.png", {1});
    scratch.write("gtFine/a.png", {1});

    MaskPNGOptions options = options_of(scratch);
    options.split.clear();
    options.image_folder = "leftImg8bit";
    options.mask_folder = "gtFine";
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "leftImg8bit/a.png");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, scratch.path()), "gtFine/a.png");
}

/** @brief Tests that an image without a mask is refused with both paths. */
TEST(Small_DatasetsMaskPng, AnImageWithoutAMaskIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_absent");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("images/train/b.jpg", {1});
    scratch.write("masks/train/a.png", {1});

    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::FileNotFound,
                      "images/train/b.jpg", "read_mask_png", "has no mask",
                      "masks/train/b.png");
}

/** @brief Tests that a mask without an image is refused with its path. */
TEST(Small_DatasetsMaskPng, AMaskWithoutAnImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_orphan");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("masks/train/a.png", {1});
    scratch.write("masks/train/z.png", {1});

    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::InvalidFile,
                      "masks/train/z.png", "read_mask_png", "the mask has no image");
}

/** @brief Tests that two images of one stem, which would share a mask, are refused. */
TEST(Small_DatasetsMaskPng, TwoImagesOfOneStemAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_shared");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("images/train/a.png", {1});
    scratch.write("masks/train/a.png", {1});

    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::InvalidFile,
                      "images/train/a.png", "shares the mask", "masks/train/a.png");
}

/** @brief Tests that a class count below 1, an absent folder and an empty one are refused. */
TEST(Small_DatasetsMaskPng, TheOptionsAndTheFoldersAreChecked)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_folders_absent");

    MaskPNGOptions options = options_of(scratch);
    options.class_count = 0;
    SA_EXPECT_FAILURE(read_mask_png(options), StatusCode::InvalidDescription, "read_mask_png",
                      "the class count is 0");

    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::FileNotFound,
                      "images/train", "no such directory");

    scratch.make_directory("images/train");
    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::FileNotFound,
                      "masks/train", "no such directory");

    scratch.make_directory("masks/train");
    SA_EXPECT_FAILURE(read_mask_png(options_of(scratch)), StatusCode::InvalidFile,
                      "images/train", "holds no image");
}

/** @brief Tests that the suffixes pair names that differ, as those of Cityscapes do. */
TEST(Small_DatasetsMaskPng, TheSuffixesPairNamesThatDiffer)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_suffixes");
    scratch.write("leftImg8bit/train/aachen/aachen_000000_leftImg8bit.png", {1});
    scratch.write("leftImg8bit/train/bonn/bonn_000001_leftImg8bit.png", {1});
    scratch.write("gtFine/train/aachen/aachen_000000_gtFine_labelIds.png", {1});
    scratch.write("gtFine/train/aachen/aachen_000000_gtFine_color.png", {1});
    scratch.write("gtFine/train/bonn/bonn_000001_gtFine_labelIds.png", {1});

    MaskPNGOptions options = options_of(scratch);
    options.image_folder = "leftImg8bit";
    options.mask_folder = "gtFine";
    options.image_suffix = "_leftImg8bit";
    options.mask_suffix = "_gtFine_labelIds";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_mask_png(options));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_EQ(list.files.size(), 4u);
    ASSERT_TRUE(list.samples[0].mask.has_value());
    ASSERT_TRUE(list.samples[1].mask.has_value());
    EXPECT_EQ(relative_file(list, list.samples[0].file, scratch.path()),
              "leftImg8bit/train/aachen/aachen_000000_leftImg8bit.png");
    EXPECT_EQ(relative_file(list, *list.samples[0].mask, scratch.path()),
              "gtFine/train/aachen/aachen_000000_gtFine_labelIds.png");
    EXPECT_EQ(relative_file(list, *list.samples[1].mask, scratch.path()),
              "gtFine/train/bonn/bonn_000001_gtFine_labelIds.png");
}

/** @brief Tests that an image whose name lacks the image suffix is refused with its path. */
TEST(Small_DatasetsMaskPng, AnImageWithoutTheImageSuffixIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_suffix_missing");
    scratch.write("images/train/a_left.png", {1});
    scratch.write("images/train/b.png", {1});
    scratch.write("masks/train/a_ids.png", {1});

    MaskPNGOptions options = options_of(scratch);
    options.image_suffix = "_left";
    options.mask_suffix = "_ids";

    SA_EXPECT_FAILURE(read_mask_png(options), StatusCode::InvalidFile, "images/train/b.png",
                      "read_mask_png", "does not end in '_left'");
}

/** @brief Tests that a mask of the mask suffix that no image is paired with is refused. */
TEST(Small_DatasetsMaskPng, AMaskOfTheSuffixWithoutAnImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_mask_suffix_unpaired");
    scratch.write("images/train/a.jpg", {1});
    scratch.write("masks/train/a_ids.png", {1});
    scratch.write("masks/train/b_ids.png", {1});

    MaskPNGOptions options = options_of(scratch);
    options.mask_suffix = "_ids";

    SA_EXPECT_FAILURE(read_mask_png(options), StatusCode::InvalidFile, "masks/train/b_ids.png",
                      "the mask has no image");
}
