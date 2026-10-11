/**************************************************************************/
/* test_datasets_idx.cpp                                                  */
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
 * @file test_datasets_idx.cpp
 * @brief Tests the reader of a dataset kept as an IDX image file and an IDX label file.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/datasets/idx.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <common/test_status.hpp>

#include "datasets_test_support.hpp"

using namespace SushiAI::Vision::Datasets;
using SushiAI::StatusCode;
using SushiAI::Vision::Imaging::Image;

namespace
{
    /** @brief Names the IDX element code of an unsigned byte. */
    constexpr std::uint8_t K_UBYTE = 0x08;

    /** @brief Returns options that read the training split of a directory. */
    IDXOptions train_options(const Support::ScratchDirectory& scratch)
    {
        IDXOptions options;
        options.root = scratch.path();
        return options;
    }

    /** @brief Writes three 2 by 3 images of rising bytes and the labels 4, 0 and 2. */
    void write_train_split(const Support::ScratchDirectory& scratch)
    {
        scratch.write("train-images-idx3-ubyte",
                      Support::idx_bytes(K_UBYTE, {3, 2, 3}, Support::rising_bytes(18, 100)));
        scratch.write("train-labels-idx1-ubyte", Support::idx_bytes(K_UBYTE, {3}, {4, 0, 2}));
    }
} // namespace

/** @brief Tests that a sample has the class of its label and the place of its pixels. */
TEST(Unit_DatasetsIdx, ASampleHasItsLabelAndThePlaceOfItsPixels)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_samples");
    write_train_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(train_options(scratch)));

    ASSERT_EQ(list.files.size(), 1u);
    EXPECT_EQ(list.files[0].filename().generic_string(), "train-images-idx3-ubyte");
    EXPECT_TRUE(list.class_names.empty());
    EXPECT_EQ(list.class_count, 5);
    ASSERT_EQ(list.samples.size(), 3u);

    const std::int32_t labels[] = {4, 0, 2};
    for (std::size_t i = 0; i < list.samples.size(); ++i)
    {
        const Sample& sample = list.samples[i];
        EXPECT_EQ(sample.file, 0u);
        EXPECT_EQ(sample.class_index, labels[i]);
        EXPECT_TRUE(sample.objects.is_empty());

        ASSERT_TRUE(sample.stored.has_value());
        EXPECT_EQ(sample.stored->offset, 16u + 6u * i);
        EXPECT_EQ(sample.stored->size.height, 2);
        EXPECT_EQ(sample.stored->size.width, 3);
        EXPECT_EQ(sample.stored->channels, 1);
        EXPECT_EQ(sample.stored->layout, PlaneLayout::Interleaved);
    }
}

/** @brief Tests that a sample's stored pixels read back as the bytes of its image. */
TEST(Unit_DatasetsIdx, ASamplesPixelsReadBackFromTheImageFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_pixels");
    write_train_split(scratch);

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(train_options(scratch)));
    ASSERT_EQ(list.samples.size(), 3u);

    Image image;
    SA_EXPECT_OK(read_stored_pixels(list.files[list.samples[2].file], *list.samples[2].stored,
                                    image));

    const std::vector<std::uint8_t> pixels(image.data(), image.data() + image.byte_count());
    EXPECT_EQ(pixels, Support::rising_bytes(6, 112));
}

/** @brief Tests that the test split reads the t10k file names. */
TEST(Unit_DatasetsIdx, TheTestSplitReadsTheT10kFiles)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_test_split");
    scratch.write("t10k-images-idx3-ubyte", Support::idx_bytes(K_UBYTE, {1, 1, 1}, {9}));
    scratch.write("t10k-labels-idx1-ubyte", Support::idx_bytes(K_UBYTE, {1}, {7}));

    IDXOptions options = train_options(scratch);
    options.split = "test";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(options));

    ASSERT_EQ(list.samples.size(), 1u);
    EXPECT_EQ(list.samples[0].class_index, 7);
    EXPECT_EQ(list.class_count, 8);
}

/** @brief Tests that named files are read whatever the split says. */
TEST(Unit_DatasetsIdx, NamedFilesAreReadForAnySplit)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_named");
    scratch.write("images.idx",
                  Support::idx_bytes(K_UBYTE, {2, 1, 1, 3}, Support::rising_bytes(6, 0)));
    scratch.write("labels.idx", Support::idx_bytes(K_UBYTE, {2}, {1, 1}));

    IDXOptions options = train_options(scratch);
    options.split = "balanced";
    options.image_file = "images.idx";
    options.label_file = "labels.idx";

    SA_ASSERT_OK_AND_ASSIGN(const SampleList list, read_idx(options));

    ASSERT_EQ(list.samples.size(), 2u);
    ASSERT_TRUE(list.samples[1].stored.has_value());
    EXPECT_EQ(list.samples[1].stored->channels, 3);
    EXPECT_EQ(list.samples[1].stored->offset, 20u + 3u);
}

/** @brief Tests that a split with no standard file name and no named file is refused. */
TEST(Unit_DatasetsIdx, ASplitWithoutAFileNameIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_split");
    IDXOptions options = train_options(scratch);
    options.split = "validation";

    SA_EXPECT_FAILURE(read_idx(options), StatusCode::InvalidDescription, "validation");
}

/** @brief Tests that an absent file of the split is reported with its path. */
TEST(Unit_DatasetsIdx, AnAbsentFileIsNotFound)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_absent");
    scratch.write("train-images-idx3-ubyte", Support::idx_bytes(K_UBYTE, {1, 1, 1}, {9}));

    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::FileNotFound,
                      "train-labels-idx1-ubyte");
}

/** @brief Tests that an image count that differs from the label count names both files. */
TEST(Unit_DatasetsIdx, CountsThatDifferAreRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_counts");
    scratch.write("train-images-idx3-ubyte",
                  Support::idx_bytes(K_UBYTE, {2, 1, 1}, Support::rising_bytes(2, 0)));
    scratch.write("train-labels-idx1-ubyte", Support::idx_bytes(K_UBYTE, {3}, {0, 1, 2}));

    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "2 images", "train-labels-idx1-ubyte",
                      "3 labels");
}

/** @brief Tests that an image file of another rank or element type is refused. */
TEST(Unit_DatasetsIdx, AnImageFileOfAnotherFormIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_form");
    scratch.write("train-labels-idx1-ubyte", Support::idx_bytes(K_UBYTE, {2}, {0, 1}));

    scratch.write("train-images-idx3-ubyte",
                  Support::idx_bytes(K_UBYTE, {2, 2}, Support::rising_bytes(4, 0)));
    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "2 dimensions");

    scratch.write("train-images-idx3-ubyte",
                  Support::idx_bytes(0x0C, {2, 1, 1}, Support::rising_bytes(8, 0)));
    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "code 12");

    scratch.write("train-images-idx3-ubyte",
                  Support::idx_bytes(K_UBYTE, {2, 1, 1, 5}, Support::rising_bytes(10, 0)));
    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::Unsupported,
                      "train-images-idx3-ubyte", "5 channels");
}

/** @brief Tests that an image file with no image is refused. */
TEST(Unit_DatasetsIdx, AnImageFileWithoutAnImageIsRefused)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_idx_empty");
    scratch.write("train-images-idx3-ubyte", Support::idx_bytes(K_UBYTE, {0, 2, 2}, {}));
    scratch.write("train-labels-idx1-ubyte", Support::idx_bytes(K_UBYTE, {0}, {}));

    SA_EXPECT_FAILURE(read_idx(train_options(scratch)), StatusCode::InvalidFile,
                      "train-images-idx3-ubyte", "no image");
}
