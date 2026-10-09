/**************************************************************************/
/* test_datasets_mnist.cpp                                                */
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
 * @file test_datasets_mnist.cpp
 * @brief Tests the "mnist" row the vision registry adds and the MNIST loader behind it.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiAI/vision/registration.hpp>
#include "../common/test_status.hpp"
#include "../common/test_throws.hpp"
#include "../common/test_utils.hpp"

using namespace SushiAI;
using Vision::Datasets::MnistLayout;
using Vision::Datasets::MnistOptions;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_VisionMnist : public SaTest::EngineTest
    {
        protected:
            /** @brief Adds the "mnist" row the vision registry holds. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                SA_EXPECT_OK(Vision::register_datasets(datasets));
            }

            /** @brief Holds the dataset rows the tests build from. */
            Data::DatasetRegistry datasets;
    };

    /** @brief Parses @p text as a bare attribute object. */
    Config::Value attributes_from(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }
} // namespace

/** @brief Tests that the dataset registry carries a row named "mnist". */
TEST_F(Integration_VisionMnist, TheDatasetRegistryListsMnist)
{
    EXPECT_NE(datasets.find("mnist"), nullptr);

    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "coco", attributes_from("{}"), "data"),
        StatusCode::InvalidDescription, "'mnist'");
}

/** @brief Tests that "mnist" fails, naming the directory and its files, when they are absent. */
TEST_F(Integration_VisionMnist, MnistRefusesADirectoryWithoutTheFiles)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "mnist",
                           attributes_from(R"({"directory": "sushiai_absent_mnist"})"),
                           "data"),
        StatusCode::FileNotFound, "data", "sushiai_absent_mnist", "MNIST train split",
        "train-images-idx3-ubyte, labels", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte");
}

/** @brief Tests that the "test" split of "mnist" looks for the t10k file pair. */
TEST_F(Integration_VisionMnist, MnistTestSplitLooksForTheT10kFiles)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "split": "test"})"),
            "data"),
        StatusCode::FileNotFound, "data", "sushiai_absent_mnist", "MNIST test split",
        "t10k-images-idx3-ubyte, labels");
}

/** @brief Tests that "mnist" throws when the required "directory" key is absent. */
TEST_F(Integration_VisionMnist, MnistRequiresADirectory)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(datasets, *engine, "mnist",
                                 attributes_from(R"({"split": "test"})"), "data"),
        "data", "has no", "'directory'");
}

/** @brief Tests that "mnist" rejects a key it does not know before it looks for any file. */
TEST_F(Integration_VisionMnist, AnUnknownMnistKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "batch_sizes": 16})"),
            "data"),
        "data", "'batch_sizes'", "'batch_size'");
}

/** @brief Tests that "mnist" rejects a split other than "train" or "test". */
TEST_F(Integration_VisionMnist, MnistRefusesASplitItDoesNotKnow)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "split": "validation"})"),
            "data"),
        StatusCode::InvalidDescription, "data", "'split'", "'validation'", "'train'", "'test'");
}

/** @brief Tests that "mnist" refuses a layout other than "flat" or "image" before any file. */
TEST_F(Integration_VisionMnist, AnMnistLayoutOtherThanFlatOrImageIsRefused)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(
            datasets, *engine, "mnist",
            attributes_from(R"({"directory": "sushiai_absent_mnist", "layout": "tiled"})"),
            "data"),
        StatusCode::InvalidDescription, "data", "'layout'", "'tiled'", "'flat'", "'image'");
}

/** @brief Tests that the "image" layout yields [N, rows, cols, 1] batches from the IDX files. */
TEST_F(Integration_VisionMnist, AnImageLayoutYieldsRankFourBatches)
{
    MnistOptions options;
    options.layout = MnistLayout::Image;
    options.batch_size = 16;
    options.max_samples = 32;
    if (!Vision::Datasets::mnist_available(options))
        GTEST_SKIP() << "MNIST IDX files are absent from " << options.directory.string();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> dataset,
                            Vision::Datasets::try_load_mnist(*engine, options));

    EXPECT_EQ(dataset->sample_shape(), (Shape{28, 28, 1}));
    EXPECT_EQ(dataset->input_dim(), 784);

    const SushiBLAS::Tensor& inputs = dataset->batch(0).inputs;
    ASSERT_EQ(inputs.rank, 4);
    EXPECT_EQ(inputs.shape[0], 16);
    EXPECT_EQ(inputs.shape[1], 28);
    EXPECT_EQ(inputs.shape[2], 28);
    EXPECT_EQ(inputs.shape[3], 1);
}

/** @brief Tests that an absent directory is unavailable, refused and named. */
TEST_F(Integration_VisionMnist, MnistFallsBackToSyntheticWhenTheFilesAreAbsent)
{
    MnistOptions options;
    options.directory = "this-directory-does-not-exist";

    EXPECT_FALSE(Vision::Datasets::mnist_available(options));
    SA_EXPECT_FAILURE(Vision::Datasets::try_load_mnist(*engine, options),
                      StatusCode::FileNotFound, "this-directory-does-not-exist");
    EXPECT_FALSE(Vision::Datasets::mnist_expected_paths(options).empty());
}
