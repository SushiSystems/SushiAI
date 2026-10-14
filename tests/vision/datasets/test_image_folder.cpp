/**************************************************************************/
/* test_image_folder.cpp                                                  */
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
 * @file test_image_folder.cpp
 * @brief Tests the reader of a dataset kept as one folder of images per class.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>
#include <SushiAI/vision/datasets/image_folder.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read one split of a directory. */
    ImageFolderOptions options_of(const Support::ScratchDirectory& scratch, const char* split)
    {
        ImageFolderOptions options;
        options.root = scratch.path();
        options.split = split;
        return options;
    }

    /** @brief Returns the path of a sample's file below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }
} // namespace

/** @brief Tests that the image extensions are matched without regard to case. */
TEST(Unit_DatasetsImageFolder, ImageExtensionsIgnoreCase)
{
    EXPECT_TRUE(has_image_extension("a.jpg"));
    EXPECT_TRUE(has_image_extension("a.JPEG"));
    EXPECT_TRUE(has_image_extension("dir/a.PNG"));
    EXPECT_TRUE(has_image_extension("a.saim"));
    EXPECT_FALSE(has_image_extension("a.txt"));
    EXPECT_FALSE(has_image_extension("a.jpg.bak"));
    EXPECT_FALSE(has_image_extension("jpg"));
}

/** @brief Tests that the classes are the folder names in sorted order. */
TEST(Unit_DatasetsImageFolder, ClassesAreTheSortedFolderNames)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_classes");
    scratch.write("zebra/1.png", {1});
    scratch.write("ant/1.jpg", {1});
    scratch.write("moth/1.jpeg", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(scratch, "")));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"ant", "moth", "zebra"}));
    EXPECT_EQ(list.class_count, 3);
    ASSERT_EQ(list.samples.size(), 3u);

    for (std::size_t i = 0; i < list.samples.size(); ++i)
    {
        EXPECT_EQ(list.samples[i].class_index, static_cast<std::int32_t>(i));
        EXPECT_FALSE(list.samples[i].stored.has_value());
        EXPECT_TRUE(list.samples[i].objects.is_empty());
    }
}

/** @brief Tests that a class lists its image files sorted by path, at any depth. */
TEST(Unit_DatasetsImageFolder, AClassListsItsImageFilesSortedAtAnyDepth)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_files");
    scratch.write("cat/b.png", {1});
    scratch.write("cat/a.JPG", {1});
    scratch.write("cat/notes.txt", {1});
    scratch.write("cat/more/c.saim", {1});
    scratch.write("dog/a.png", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(scratch, "")));

    ASSERT_EQ(list.samples.size(), 4u);
    EXPECT_EQ(list.files.size(), 4u);
    EXPECT_EQ(relative_file(list, 0, scratch.path()), "cat/a.JPG");
    EXPECT_EQ(relative_file(list, 1, scratch.path()), "cat/b.png");
    EXPECT_EQ(relative_file(list, 2, scratch.path()), "cat/more/c.saim");
    EXPECT_EQ(relative_file(list, 3, scratch.path()), "dog/a.png");
    EXPECT_EQ(list.samples[2].class_index, 0);
    EXPECT_EQ(list.samples[3].class_index, 1);
}

/** @brief Tests that a split names a folder of the root, and a file there is no class. */
TEST(Unit_DatasetsImageFolder, ASplitNamesAFolderOfTheRoot)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_split");
    scratch.write("train/no/1.png", {1});
    scratch.write("val/no/1.png", {1});
    scratch.write("val/yes/1.png", {1});
    scratch.write("val/stray.png", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(scratch, "val")));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"no", "yes"}));
    EXPECT_EQ(list.samples.size(), 2u);
}

/** @brief Tests that a folder without an image is still a class, so indices stay put. */
TEST(Unit_DatasetsImageFolder, AFolderWithoutAnImageIsStillAClass)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_empty_class");
    scratch.make_directory("a");
    scratch.write("b/1.png", {1});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(scratch, "")));

    EXPECT_EQ(list.class_count, 2);
    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(list.samples[0].class_index, 1);
}

/** @brief Tests that a split that is not a directory is reported as absent. */
TEST(Unit_DatasetsImageFolder, AnAbsentSplitIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_absent");

    SA_EXPECT_FAILURE(read_image_folder(options_of(scratch, "train")), StatusCode::FileNotFound,
                      "train");
}

/** @brief Tests that a split with no class folder, or with no image, is refused. */
TEST(Unit_DatasetsImageFolder, ASplitWithoutAClassOrAnImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_nothing");
    scratch.write("flat/1.png", {1});
    scratch.write("text/a/readme.txt", {1});

    SA_EXPECT_FAILURE(read_image_folder(options_of(scratch, "flat")), StatusCode::InvalidFile,
                      "flat", "no class folder");
    SA_EXPECT_FAILURE(read_image_folder(options_of(scratch, "text")), StatusCode::InvalidFile,
                      "text", "no image file");
}
