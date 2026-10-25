/**************************************************************************/
/* test_mnist.cpp                                                         */
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
 * @file test_mnist.cpp
 * @brief Tests the MNIST loader over tiny IDX files written by the test.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "datasets_test_support.hpp"
#include <SushiAI/config/value.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Bounds the error of a pixel the loader scales and the test scales the same way. */
    constexpr float K_PIXEL_TOLERANCE = 0.0f;

    /** @brief Holds the IDX code of unsigned bytes. */
    constexpr std::uint8_t K_UNSIGNED_BYTE = 0x08;

    /** @brief Names the IDX image file of the training split. */
    constexpr const char* K_IMAGE_FILE = "train-images-idx3-ubyte";

    /** @brief Holds the class of each of the five images the tiny split holds. */
    const std::vector<std::uint8_t> K_LABELS = {3, 1, 4, 1, 5};

    /** @brief Provides an engine and a training split of five 2 by 3 images. */
    class Medium_DatasetsMNIST : public SaTest::EngineTest
    {
        protected:
            /** @brief Writes the image and label files; pixel j of image i holds 6 i + j. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                scratch.write(K_IMAGE_FILE, Support::idx_bytes(K_UNSIGNED_BYTE, {5, 2, 3},
                                                               Support::rising_bytes(30, 0)));
                scratch.write("train-labels-idx1-ubyte",
                              Support::idx_bytes(K_UNSIGNED_BYTE, {5}, K_LABELS));
            }

            /** @brief Returns options over the scratch directory, batches of two, every sample. */
            MNISTOptions options() const
            {
                MNISTOptions made;
                made.directory = scratch.path();
                made.batch_size = 2;
                made.max_samples = 0;
                return made;
            }

            /** @brief Owns the directory the IDX files are written to. */
            const Support::ScratchDirectory scratch{"sushiai_test_datasets_mnist"};
    };
} // namespace

/** @brief Tests that the defaults read a capped training split in rows of pixels. */
TEST(Small_DatasetsMNIST, TheDefaultsReadACappedTrainingSplitAsRows)
{
    const MNISTOptions options;

    EXPECT_EQ(options.directory, "data/mnist");
    EXPECT_EQ(options.batch_size, 64);
    EXPECT_EQ(options.max_samples, 4096);
    EXPECT_FALSE(options.test_split);
    EXPECT_EQ(options.layout, MNISTLayout::Flat);
}

/** @brief Tests that five samples load as two whole batches of scaled pixels and labels. */
TEST_F(Medium_DatasetsMNIST, TheSamplesLoadAsWholeBatches)
{
    ASSERT_TRUE(mnist_available(options()));
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            try_load_mnist(*engine, options()));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->sample_shape(), Shape{6});

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));
    const float* pixels = batch.inputs.data_as<float>();
    for (std::size_t i = 0; i < 12; ++i)
        EXPECT_NEAR(pixels[i], static_cast<float>(12 + i) / 255.0f, K_PIXEL_TOLERANCE) << i;
    const std::int32_t* labels = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(labels[0], 4);
    EXPECT_EQ(labels[1], 1);
}

/** @brief Tests that the cap on samples leaves out the samples past it. */
TEST_F(Medium_DatasetsMNIST, TheSampleCapLimitsTheBatches)
{
    MNISTOptions capped = options();
    capped.max_samples = 3;

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            try_load_mnist(*engine, capped));

    EXPECT_EQ(dataset->batch_count(), 1u);
}

/** @brief Tests that a batch size that is not positive is refused. */
TEST_F(Medium_DatasetsMNIST, ABatchSizeThatIsNotPositiveIsRefused)
{
    MNISTOptions zero = options();
    zero.batch_size = 0;

    SA_EXPECT_FAILURE(try_load_mnist(*engine, zero), StatusCode::InvalidDescription,
                      "positive batch size");
}

/** @brief Tests that an image file holding a label file's header is refused as invalid. */
TEST_F(Medium_DatasetsMNIST, AnImageFileWithoutTheIDX3HeaderIsInvalid)
{
    scratch.write(K_IMAGE_FILE, Support::idx_bytes(K_UNSIGNED_BYTE, {5}, K_LABELS));

    SA_EXPECT_FAILURE(try_load_mnist(*engine, options()), StatusCode::InvalidFile,
                      "is not an IDX3 image file");
}

/** @brief Tests that the factory reads the image layout over the files of the directory. */
TEST_F(Medium_DatasetsMNIST, TheFactoryReadsTheImageLayout)
{
    const Config::Value attributes = Config::parse_json(
        "{\"directory\": \"" + scratch.path().generic_string() +
            "\", \"batch_size\": 2, \"max_samples\": 0, \"layout\": \"image\"}",
        "<test>");

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            make_mnist(*engine, attributes, "data"));

    EXPECT_EQ(dataset->sample_shape(), (Shape{2, 3, 1}));
    EXPECT_EQ(dataset->batch_count(), 2u);
}
