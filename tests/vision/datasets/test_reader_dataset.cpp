/**************************************************************************/
/* test_reader_dataset.cpp                                                */
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
 * @file test_reader_dataset.cpp
 * @brief Tests the dataset that builds each batch of a sample list when it is asked for.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/SushiAI.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/reader_dataset.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::NormalizedBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the absolute tolerance of a byte divided by 255 in float. */
    constexpr float K_SCALE_TOLERANCE = 1e-6f;

    /** @brief Holds the absolute tolerance of a box coordinate, in pixels. */
    constexpr float K_PIXEL_TOLERANCE = 1e-4f;

    /** @brief Provides an engine and the builders of the lists these tests read. */
    class Unit_DatasetsReaderDataset : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Returns a list of 2 by 2 gray samples stored in one file of rising bytes.
             * @param count Samples; sample i has the class i modulo 2.
             */
            SampleList stored_list(const Support::ScratchDirectory& scratch, std::size_t count)
            {
                SampleList list;
                list.files.push_back(
                    scratch.write("pixels.bin", Support::rising_bytes(count * 4, 0)));
                list.class_count = 2;

                for (std::size_t i = 0; i < count; ++i)
                {
                    StoredPixels stored;
                    stored.offset = i * 4;
                    stored.size = ImageSize{2, 2};
                    stored.channels = 1;

                    Sample sample;
                    sample.stored = stored;
                    sample.class_index = static_cast<std::int32_t>(i % 2);
                    list.samples.push_back(sample);
                }

                return list;
            }

            /** @brief Returns the options of batches of two gray 2 by 2 inputs. */
            ReaderDatasetOptions gray_options() const
            {
                ReaderDatasetOptions options;
                options.fit.size = ImageSize{2, 2};
                options.channels = 1;
                options.batch_size = 2;
                options.name = "tiny";
                return options;
            }

            /** @brief Returns a classification collator over two classes. */
            std::unique_ptr<const Collator> classification()
            {
                Result<std::unique_ptr<ClassificationCollator>> made =
                    ClassificationCollator::make(2);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }
    };
} // namespace

/** @brief Tests that the dataset reports the batch size, the shapes and the whole batches. */
TEST_F(Unit_DatasetsReaderDataset, ItDescribesItsBatches)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_describe");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 5), classification(), gray_options()));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->sample_shape(), (Shape{2, 2, 1}));
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->name(), "tiny");
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_LABELS);
}

/** @brief Tests that batch i holds samples 2i and 2i + 1, scaled, with their labels. */
TEST_F(Unit_DatasetsReaderDataset, ABatchHoldsItsSamplesInListOrder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_order");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 4), classification(), gray_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    const float* inputs = batch.inputs.data_as<float>();
    for (std::size_t i = 0; i < 8; ++i)
        EXPECT_NEAR(inputs[i], static_cast<float>(8 + i) / 255.0f, K_SCALE_TOLERANCE) << i;

    ASSERT_EQ(batch.targets.size(), 1u);
    const std::int32_t* labels = batch.targets[0].values.data_as<std::int32_t>();
    EXPECT_EQ(labels[0], 0);
    EXPECT_EQ(labels[1], 1);
}

/** @brief Tests that a detection batch letterboxes each file and moves its boxes with it. */
TEST_F(Unit_DatasetsReaderDataset, ADetectionBatchLetterboxesImagesAndBoxes)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_detection");

    SampleList list;
    list.class_count = 3;
    for (const char* name : {"a.saim", "b.saim"})
    {
        const std::filesystem::path file = scratch.path() / name;
        Vision::Imaging::write_raw_image_file(
            file, Support::uniform_image(ImageSize{2, 4}, 3, 255).view(),
            Vision::Imaging::ChannelOrder::RGB);

        Sample sample;
        sample.file = static_cast<std::uint32_t>(list.files.size());
        list.files.push_back(file);
        list.samples.push_back(sample);
    }
    list.samples[1].objects.add(2, NormalizedBox{0.5f, 0.5f, 0.5f, 1.0f});

    ReaderDatasetOptions options;
    options.fit.size = ImageSize{4, 4};
    options.fit.fit = SampleFit::Letterbox;
    options.batch_size = 2;

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<const Collator> collator,
                            DetectionCollator::make(3, 2));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, std::move(list), std::move(collator), options));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    // Row 0 of each 4 by 4 by 3 input is padding of 114; row 1 is the image.
    const float* inputs = batch.inputs.data_as<float>();
    EXPECT_NEAR(inputs[0], 114.0f / 255.0f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[12], 1.0f, K_SCALE_TOLERANCE);

    ASSERT_EQ(batch.targets.size(), 3u);
    const float* boxes = batch.targets[0].values.data_as<float>();
    const std::int32_t* classes = batch.targets[1].values.data_as<std::int32_t>();
    const float* valid = batch.targets[2].values.data_as<float>();

    EXPECT_EQ(valid[0], 0.0f);
    EXPECT_EQ(valid[2], 1.0f);
    EXPECT_EQ(valid[3], 0.0f);
    EXPECT_EQ(classes[2], 2);
    EXPECT_NEAR(boxes[8], 1.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[9], 1.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[10], 3.0f, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[11], 3.0f, K_PIXEL_TOLERANCE);
}

/** @brief Tests that make refuses options no batch can be built from. */
TEST_F(Unit_DatasetsReaderDataset, MakeRefusesOptionsOutsideTheirBounds)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_refusals");
    const SampleList list = stored_list(scratch, 2);

    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, list, nullptr, gray_options()),
                      StatusCode::InvalidDescription, "ReaderDataset", "tiny", "no collator");

    ReaderDatasetOptions options = gray_options();
    options.fit.size = ImageSize{0, 2};
    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, list, classification(), options),
                      StatusCode::InvalidDescription, "0 by 2");

    options = gray_options();
    options.channels = 5;
    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, list, classification(), options),
                      StatusCode::InvalidDescription, "5 channels");

    options = gray_options();
    options.batch_size = 0;
    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, list, classification(), options),
                      StatusCode::InvalidDescription, "batch size of 0");

    options = gray_options();
    options.batch_size = 3;
    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, list, classification(), options),
                      StatusCode::InvalidDescription, "2 samples", "one batch of 3");
}

/** @brief Tests that a batch index past the last batch stops the process. */
TEST_F(Unit_DatasetsReaderDataset, ABatchOutsideTheEpochStopsTheProcess)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_index");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 2), classification(), gray_options()));

    Data::Batch batch;
    SA_EXPECT_FATAL((void)dataset->read(1, batch), "ReaderDataset", "1 batches", "batch 1");
}

/** @brief Tests that a second read into the same batch writes the tensors of the first. */
TEST_F(Unit_DatasetsReaderDataset, AReadReusesTheTensorsItFilledBefore)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_reuse");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 4), classification(), gray_options()));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const float* const first = batch.inputs.data_as<float>();
    const std::int32_t* const first_labels = batch.targets[0].values.data_as<std::int32_t>();

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));
    EXPECT_EQ(batch.inputs.data_as<float>(), first);
    EXPECT_EQ(batch.targets[0].values.data_as<std::int32_t>(), first_labels);
    EXPECT_NEAR(first[0], 8.0f / 255.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that a read leaves alone a tensor something else still references. */
TEST_F(Unit_DatasetsReaderDataset, AReadDoesNotWriteATensorSomethingElseHolds)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_shared");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 4), classification(), gray_options()));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const SushiBLAS::Tensor kept = batch.inputs;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));
    EXPECT_NE(batch.inputs.data_as<float>(), kept.data_as<float>());
    EXPECT_NEAR(kept.data_as<float>()[0], 0.0f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(batch.inputs.data_as<float>()[0], 8.0f / 255.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that a reader dataset is read through the Data::Dataset interface. */
TEST_F(Unit_DatasetsReaderDataset, ItIsADatasetATrainerCanRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_seam");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> made,
        ReaderDataset::make(*engine, stored_list(scratch, 4), classification(), gray_options()));
    const Data::Dataset& dataset = *made;

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, batch)));
    ASSERT_EQ(batch.targets.size(), dataset.targets().size());
    EXPECT_EQ(batch.targets[0].role, dataset.targets()[0].role);
    EXPECT_EQ(batch.inputs.shape[0], dataset.batch_size());
}

/** @brief Tests that an image of another channel count is a failure naming its file. */
TEST_F(Unit_DatasetsReaderDataset, AnImageOfAnotherChannelCountNamesItsFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_channels");

    ReaderDatasetOptions options = gray_options();
    options.channels = 3;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 2), classification(), options));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::Incompatible, "pixels.bin",
                      "ReaderDataset", "1 channels", "reads 3");
}

/** @brief Tests that a file that is absent when its batch is read is a failure naming it. */
TEST_F(Unit_DatasetsReaderDataset, AFileAbsentAtReadTimeIsNamed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_absent");

    SampleList list = stored_list(scratch, 2);
    list.files[0] = scratch.path() / "gone.bin";

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, std::move(list), classification(), gray_options()));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::InvalidFile, "gone.bin");
}

namespace
{
    /** @brief Holds the seed of the shuffled datasets below. */
    constexpr std::uint64_t K_ORDER_SEED = 42;

    /**
     * @brief Reads every batch and returns the list index of the sample in each row.
     *
     * Sample i of stored_list opens with the byte 4i, which the input holds as 4i / 255.
     */
    std::vector<std::uint64_t> samples_read(const Data::Dataset& dataset)
    {
        std::vector<std::uint64_t> samples;
        Data::Batch batch;

        for (std::size_t index = 0; index < dataset.batch_count(); ++index)
        {
            const Status status = dataset.read(index, batch);
            EXPECT_TRUE(status.is_ok()) << status.message();
            if (!status.is_ok())
                return samples;

            const float* inputs = batch.inputs.data_as<float>();
            for (std::int64_t row = 0; row < dataset.batch_size(); ++row)
                samples.push_back(
                    static_cast<std::uint64_t>(inputs[row * 4] * 255.0f + 0.5f) / 4);
        }

        return samples;
    }

    /** @brief Returns the order Data::EpochPermutation gives a seed, an epoch and a count. */
    std::vector<std::uint64_t> order_of(std::uint64_t seed, std::int64_t epoch,
                                        std::uint64_t count)
    {
        const Data::EpochPermutation permutation(seed, epoch, count);

        std::vector<std::uint64_t> order;
        for (std::uint64_t position = 0; position < count; ++position)
            order.push_back(permutation.at(position));

        return order;
    }
} // namespace

/** @brief Tests that a shuffled epoch reads every sample once, in the order of its seed. */
TEST_F(Unit_DatasetsReaderDataset, AShuffledEpochReadsEverySampleOnceInTheOrderOfTheSeed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_shuffle");

    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 8), classification(), options));

    const std::vector<std::uint64_t> first = samples_read(*dataset);
    EXPECT_EQ(first, order_of(K_ORDER_SEED, 0, 8));

    std::vector<bool> is_read(8, false);
    for (std::uint64_t sample : first)
    {
        ASSERT_LT(sample, 8u);
        EXPECT_FALSE(is_read[sample]) << "sample " << sample;
        is_read[sample] = true;
    }
}

/** @brief Tests that begin_epoch gives the order of that epoch, and the first one again. */
TEST_F(Unit_DatasetsReaderDataset, BeginEpochSelectsTheOrderOfTheEpoch)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_epochs");

    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 8), classification(), options));

    Data::begin_epoch(*dataset, 3);
    const std::vector<std::uint64_t> third = samples_read(*dataset);
    EXPECT_EQ(third, order_of(K_ORDER_SEED, 3, 8));
    EXPECT_NE(third, order_of(K_ORDER_SEED, 0, 8));

    Data::begin_epoch(*dataset, 0);
    EXPECT_EQ(samples_read(*dataset), order_of(K_ORDER_SEED, 0, 8));
}

/** @brief Tests that without shuffle every epoch reads the samples in the list's order. */
TEST_F(Unit_DatasetsReaderDataset, WithoutShuffleEveryEpochKeepsTheOrderOfTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_unshuffled");

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<ReaderDataset> dataset,
        ReaderDataset::make(*engine, stored_list(scratch, 6), classification(), gray_options()));

    const std::vector<std::uint64_t> in_order = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(samples_read(*dataset), in_order);

    Data::begin_epoch(*dataset, 5);
    EXPECT_EQ(samples_read(*dataset), in_order);
}

/** @brief Tests that the form of make that takes a pipeline refuses a null one. */
TEST_F(Unit_DatasetsReaderDataset, MakeRefusesANullPipeline)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_pipeline");

    SA_EXPECT_FAILURE(ReaderDataset::make(*engine, stored_list(scratch, 2), classification(),
                                          nullptr, gray_options()),
                      StatusCode::InvalidDescription, "ReaderDataset", "tiny", "no pipeline");
}
