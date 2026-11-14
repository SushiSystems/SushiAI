/**************************************************************************/
/* test_image_file.cpp                                                    */
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
 * @file test_image_file.cpp
 * @brief Tests the reader of a dataset that is one image file.
 * @author Mustafa Garip
 */
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/image_file.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;

namespace
{
    /** @brief Returns options that read the file below the folder data for 80 classes. */
    ImageFileOptions options_of(const char* file)
    {
        ImageFileOptions options;
        options.root = "data";
        options.file = file;
        options.class_count = 80;
        return options;
    }
} // namespace

/** @brief Tests that the image becomes the one sample of the list, with no object and no class. */
TEST(Small_DatasetsImageFile, TheImageIsTheOneSampleWithNoObjectAndNoClass)
{
    MemorySource source;
    source.add_file("data/cat.JPG", "x");

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_image_file(options_of("cat.JPG"), source));

    ASSERT_EQ(list.samples.size(), 1u);
    ASSERT_EQ(list.files.size(), 1u);
    EXPECT_EQ(list.files[list.samples[0].file].generic_string(), "data/cat.JPG");
    EXPECT_TRUE(list.samples[0].objects.is_empty());
    EXPECT_FALSE(list.samples[0].class_index.has_value());
    EXPECT_EQ(list.class_count, 80);
}

/** @brief Tests that a file with no image extension is refused. */
TEST(Small_DatasetsImageFile, AFileWithNoImageExtensionIsRefused)
{
    MemorySource source;
    source.add_file("data/notes.txt", "x");

    SA_EXPECT_FAILURE(read_image_file(options_of("notes.txt"), source),
                      StatusCode::InvalidDescription, "no image extension");
}

/** @brief Tests that a file that does not exist is not found. */
TEST(Small_DatasetsImageFile, AFileThatIsAbsentIsNotFound)
{
    const MemorySource source;

    SA_EXPECT_FAILURE(read_image_file(options_of("absent.png"), source), StatusCode::FileNotFound,
                      "absent.png");
}

/** @brief Tests that a class count below 1 is refused. */
TEST(Small_DatasetsImageFile, ANoClassCountIsRefused)
{
    MemorySource source;
    source.add_file("data/cat.png", "x");
    ImageFileOptions options = options_of("cat.png");
    options.class_count = 0;

    SA_EXPECT_FAILURE(read_image_file(options, source), StatusCode::InvalidDescription,
                      "class count");
}
