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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/image_folder.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read one split of the folder data. */
    ImageFolderOptions options_of(const char* split)
    {
        ImageFolderOptions options;
        options.root = "data";
        options.split = split;
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

    /** @brief Returns the path of a sample's file below a directory, with forward slashes. */
    std::string relative_file(const SampleList& list, std::size_t sample,
                              const std::filesystem::path& base)
    {
        return list.files[list.samples[sample].file].lexically_relative(base).generic_string();
    }

    /** @brief Returns the class index of each sample of a list, in order. */
    std::vector<std::int32_t> classes_of(const SampleList& list)
    {
        std::vector<std::int32_t> classes;
        for (const Sample& sample : list.samples)
            classes.push_back(sample.class_index.value_or(-1));

        return classes;
    }
} // namespace

/** @brief Tests that the classes are the folder names in sorted order. */
TEST(Small_DatasetsImageFolder, ClassesAreTheSortedFolderNames)
{
    const MemorySource source = source_of({"zebra/1.png", "ant/1.jpg", "moth/1.jpeg"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(""), source));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"ant", "moth", "zebra"}));
    EXPECT_EQ(list.class_count, 3);
    EXPECT_EQ(classes_of(list), (std::vector<std::int32_t>{0, 1, 2}));
}

/** @brief Tests that a listed sample names no stored pixels and no object. */
TEST(Small_DatasetsImageFolder, ASampleNamesNoStoredPixelsAndNoObject)
{
    const MemorySource source = source_of({"ant/1.jpg"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(""), source));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_FALSE(list.samples[0].stored.has_value());
    EXPECT_TRUE(list.samples[0].objects.is_empty());
}

/** @brief Tests that a class lists its image files sorted by path, at any depth. */
TEST(Small_DatasetsImageFolder, AClassListsItsImageFilesSortedAtAnyDepth)
{
    const MemorySource source = source_of(
        {"cat/b.png", "cat/a.JPG", "cat/notes.txt", "cat/more/c.saim", "dog/a.png"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(""), source));

    ASSERT_EQ(list.samples.size(), 4u);
    EXPECT_EQ(list.files.size(), 4u);
    EXPECT_EQ(relative_file(list, 0, "data"), "cat/a.JPG");
    EXPECT_EQ(relative_file(list, 1, "data"), "cat/b.png");
    EXPECT_EQ(relative_file(list, 2, "data"), "cat/more/c.saim");
    EXPECT_EQ(relative_file(list, 3, "data"), "dog/a.png");
    EXPECT_EQ(classes_of(list), (std::vector<std::int32_t>{0, 0, 0, 1}));
}

/** @brief Tests that a split names a folder of the root. */
TEST(Small_DatasetsImageFolder, ASplitNamesAFolderOfTheRoot)
{
    const MemorySource source = source_of({"train/no/1.png", "val/no/1.png", "val/yes/1.png"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of("val"), source));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"no", "yes"}));
    EXPECT_EQ(list.samples.size(), 2u);
}

/** @brief Tests that a file directly in the split is no class. */
TEST(Small_DatasetsImageFolder, AFileDirectlyInTheSplitIsNoClass)
{
    const MemorySource source = source_of({"val/no/1.png", "val/stray.png"});

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of("val"), source));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"no"}));
    EXPECT_EQ(list.samples.size(), 1u);
}

/** @brief Tests that a folder without an image is still a class, so indices stay put. */
TEST(Small_DatasetsImageFolder, AFolderWithoutAnImageIsStillAClass)
{
    MemorySource source = source_of({"b/1.png"});
    source.add_directory("data/a");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options_of(""), source));

    EXPECT_EQ(list.class_count, 2);
    EXPECT_EQ(classes_of(list), (std::vector<std::int32_t>{1}));
}

/** @brief Tests that a split with no class folder is refused. */
TEST(Small_DatasetsImageFolder, ASplitWithoutAClassFolderIsRefused)
{
    const MemorySource source = source_of({"flat/1.png"});

    SA_EXPECT_FAILURE(read_image_folder(options_of("flat"), source), StatusCode::InvalidFile,
                      "flat", "no class folder");
}

/** @brief Tests that a split whose class folders hold no image is refused. */
TEST(Small_DatasetsImageFolder, ASplitWithoutAnImageIsRefused)
{
    const MemorySource source = source_of({"text/a/readme.txt"});

    SA_EXPECT_FAILURE(read_image_folder(options_of("text"), source), StatusCode::InvalidFile,
                      "text", "no image file");
}

/** @brief Tests that a split that is not a directory is reported as absent. */
TEST(Medium_DatasetsImageFolder, AnAbsentSplitIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_absent");
    ImageFolderOptions options;
    options.root = scratch.path();
    options.split = "train";

    SA_EXPECT_FAILURE(read_image_folder(options), StatusCode::FileNotFound, "train");
}

/** @brief Tests that the file-system form lists the class folders written to disk. */
TEST(Medium_DatasetsImageFolder, ClassFoldersOnDiskBecomeSortedClasses)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_folder_disk");
    scratch.write("dog/2.png", {1});
    scratch.write("dog/deeper/3.jpg", {1});
    scratch.write("cat/1.png", {1});
    ImageFolderOptions options;
    options.root = scratch.path();

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_folder(options));

    EXPECT_EQ(list.class_names, (std::vector<std::string>{"cat", "dog"}));
    EXPECT_EQ(classes_of(list), (std::vector<std::int32_t>{0, 1, 1}));
}
