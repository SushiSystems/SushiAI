/**************************************************************************/
/* test_idx.cpp                                                           */
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
 * @file test_idx.cpp
 * @brief Tests the reader of a dataset kept as an IDX image file and an IDX label file.
 * @author Mustafa Garip
 */
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/byte_source.hpp>
#include <SushiAI/vision/datasets/idx.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/imaging/image.hpp>

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;

namespace
{
    /** @brief Names the IDX element code of an unsigned byte. */
    constexpr std::uint8_t K_UBYTE = 0x08;

    /** @brief Names the file of the training images. */
    constexpr const char* K_TRAIN_IMAGES = "data/train-images-idx3-ubyte";

    /** @brief Names the file of the training labels. */
    constexpr const char* K_TRAIN_LABELS = "data/train-labels-idx1-ubyte";

    /** @brief Returns the bytes of three 2 by 3 images of rising bytes. */
    std::vector<std::uint8_t> train_image_bytes()
    {
        return Support::idx_bytes(K_UBYTE, {3, 2, 3}, Support::rising_bytes(18, 100));
    }

    /** @brief Returns options that read the training split of the folder data. */
    IDXOptions train_options()
    {
        IDXOptions options;
        options.root = "data";
        return options;
    }

    /** @brief Returns a source holding three 2 by 3 images and the labels 4, 0 and 2. */
    MemorySource train_split()
    {
        MemorySource source;
        source.add_file(K_TRAIN_IMAGES, Support::text_of(train_image_bytes()));
        source.add_file(K_TRAIN_LABELS,
                        Support::text_of(Support::idx_bytes(K_UBYTE, {3}, {4, 0, 2})));
        return source;
    }

    /** @brief Returns a source holding the training files with the given contents. */
    MemorySource train_files(const std::vector<std::uint8_t>& images,
                             const std::vector<std::uint8_t>& labels)
    {
        MemorySource source;
        source.add_file(K_TRAIN_IMAGES, Support::text_of(images));
        source.add_file(K_TRAIN_LABELS, Support::text_of(labels));
        return source;
    }

    /** @brief Holds an image file of a form read_idx refuses, with the status and the cause. */
    struct FormRow
    {
        std::string name;                 /**< The row's name. */
        std::vector<std::uint8_t> images; /**< The bytes of the image file. */
        StatusCode code;                  /**< The status the refusal carries. */
        std::string cause;                /**< The text the message holds besides the file. */
    };

    /** @brief Runs read_idx over each image file of a form it refuses. */
    class Small_DatasetsIdxForm : public ::testing::TestWithParam<FormRow>
    {
    };

    /** @brief Names a test after its row. */
    std::string form_row_name(const ::testing::TestParamInfo<FormRow>& info)
    {
        return info.param.name;
    }
} // namespace

/** @brief Tests that a sample has the class its label file gives. */
TEST(Small_DatasetsIdx, ASampleHasTheClassOfItsLabel)
{
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(train_options(), train_split()));

    ASSERT_EQ(list.samples.size(), 3u);
    EXPECT_EQ(list.samples[0].class_index, 4);
    EXPECT_EQ(list.samples[1].class_index, 0);
    EXPECT_EQ(list.samples[2].class_index, 2);
    EXPECT_EQ(list.class_count, 5);
    EXPECT_TRUE(list.class_names.empty());
}

/** @brief Tests that a sample refers to its pixels inside the image file. */
TEST(Small_DatasetsIdx, ASampleRefersToThePlaceOfItsPixels)
{
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(train_options(), train_split()));

    ASSERT_EQ(list.files.size(), 1u);
    EXPECT_EQ(list.files[0].filename().generic_string(), "train-images-idx3-ubyte");
    ASSERT_EQ(list.samples.size(), 3u);

    std::vector<std::uint64_t> offsets;
    for (const Sample& sample : list.samples)
    {
        ASSERT_TRUE(sample.stored.has_value());
        offsets.push_back(sample.stored->offset);
        EXPECT_EQ(sample.file, 0u);
        EXPECT_EQ(sample.stored->size.height, 2);
        EXPECT_EQ(sample.stored->size.width, 3);
        EXPECT_EQ(sample.stored->channels, 1);
        EXPECT_EQ(sample.stored->layout, PlaneLayout::Interleaved);
        EXPECT_TRUE(sample.objects.is_empty());
    }

    EXPECT_EQ(offsets, (std::vector<std::uint64_t>{16, 22, 28}));
}

/** @brief Tests that a sample's stored pixels read back as the bytes of its image. */
TEST(Small_DatasetsIdx, ASamplesPixelsReadBackFromTheImageFile)
{
    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(train_options(), train_split()));
    ASSERT_EQ(list.samples.size(), 3u);
    std::istringstream file(Support::text_of(train_image_bytes()), std::ios::binary);

    Image image;
    SA_EXPECT_OK(read_stored_pixels(file, *list.samples[2].stored, image));

    const std::vector<std::uint8_t> pixels(image.data(), image.data() + image.byte_count());
    EXPECT_EQ(pixels, Support::rising_bytes(6, 112));
}

/** @brief Tests that the test split reads the t10k file names. */
TEST(Small_DatasetsIdx, TheTestSplitReadsTheT10kFiles)
{
    MemorySource source;
    source.add_file("data/t10k-images-idx3-ubyte",
                    Support::text_of(Support::idx_bytes(K_UBYTE, {1, 1, 1}, {9})));
    source.add_file("data/t10k-labels-idx1-ubyte",
                    Support::text_of(Support::idx_bytes(K_UBYTE, {1}, {7})));
    IDXOptions options = train_options();
    options.split = "test";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(options, source));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(list.samples[0].class_index, 7);
    EXPECT_EQ(list.class_count, 8);
}

/** @brief Tests that named files are read whatever the split says. */
TEST(Small_DatasetsIdx, NamedFilesAreReadForAnySplit)
{
    MemorySource source;
    source.add_file("data/images.idx", Support::text_of(Support::idx_bytes(
                                           K_UBYTE, {2, 1, 1, 3}, Support::rising_bytes(6, 0))));
    source.add_file("data/labels.idx", Support::text_of(Support::idx_bytes(K_UBYTE, {2}, {1, 1})));
    IDXOptions options = train_options();
    options.split = "balanced";
    options.image_file = "images.idx";
    options.label_file = "labels.idx";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(options, source));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_TRUE(list.samples[1].stored.has_value());
    EXPECT_EQ(list.samples[1].stored->channels, 3);
    EXPECT_EQ(list.samples[1].stored->offset, 20u + 3u);
}

/** @brief Tests that a split with no standard file name and no named file is refused. */
TEST(Small_DatasetsIdx, ASplitWithoutAFileNameIsRefused)
{
    const MemorySource source;
    IDXOptions options = train_options();
    options.split = "validation";

    SA_EXPECT_FAILURE(read_idx(options, source), StatusCode::InvalidDescription, "validation");
}

/** @brief Tests that an absent file of the split is reported with its path. */
TEST(Medium_DatasetsIdx, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_absent");
    scratch.write("train-images-idx3-ubyte", Support::idx_bytes(K_UBYTE, {1, 1, 1}, {9}));
    IDXOptions options;
    options.root = scratch.path();

    SA_EXPECT_FAILURE(read_idx(options), StatusCode::FileNotFound, "train-labels-idx1-ubyte");
}

/** @brief Tests that an image count that differs from the label count names both files. */
TEST(Small_DatasetsIdx, CountsThatDifferAreRefused)
{
    const MemorySource source =
        train_files(Support::idx_bytes(K_UBYTE, {2, 1, 1}, Support::rising_bytes(2, 0)),
                    Support::idx_bytes(K_UBYTE, {3}, {0, 1, 2}));

    SA_EXPECT_FAILURE(read_idx(train_options(), source), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "2 images", "train-labels-idx1-ubyte",
                      "3 labels");
}

/** @brief Tests that an image file of another rank, element type or channel count is refused. */
TEST_P(Small_DatasetsIdxForm, AnImageFileOfAnotherFormIsRefused)
{
    const MemorySource source =
        train_files(GetParam().images, Support::idx_bytes(K_UBYTE, {2}, {0, 1}));

    SA_EXPECT_FAILURE(read_idx(train_options(), source), GetParam().code,
                      "train-images-idx3-ubyte", GetParam().cause);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsIdxForm,
    ::testing::Values(
        FormRow{"AnotherRank", Support::idx_bytes(K_UBYTE, {2, 2}, Support::rising_bytes(4, 0)),
                StatusCode::InvalidFile, "2 dimensions"},
        FormRow{"AnotherElementType",
                Support::idx_bytes(0x0C, {2, 1, 1}, Support::rising_bytes(8, 0)),
                StatusCode::InvalidFile, "code 12"},
        FormRow{"AnotherChannelCount",
                Support::idx_bytes(K_UBYTE, {2, 1, 1, 5}, Support::rising_bytes(10, 0)),
                StatusCode::Unsupported, "5 channels"}),
    form_row_name);

/** @brief Tests that an image file with no image is refused. */
TEST(Small_DatasetsIdx, AnImageFileWithoutAnImageIsRefused)
{
    const MemorySource source = train_files(Support::idx_bytes(K_UBYTE, {0, 2, 2}, {}),
                                            Support::idx_bytes(K_UBYTE, {0}, {}));

    SA_EXPECT_FAILURE(read_idx(train_options(), source), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "no image");
}
