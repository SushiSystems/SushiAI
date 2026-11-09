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
 * @file test_reader_dataset.cpp
 * @brief Tests the dataset that builds each batch of a sample list when it is asked for.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include "datasets_test_support.hpp"
#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/reader_dataset.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
using Vision::Imaging::ImageSize;
using Vision::Imaging::NormalizedBox;
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Holds the absolute tolerance of a byte divided by 255 in float: 32 times the float
     *        unit roundoff, the smallest power of two not tighter than the earlier 1e-6f.
     */
    constexpr float K_SCALE_TOLERANCE =
        static_cast<float>(32.0 * TestSupport::unit_roundoff<float>());

    /**
     * @brief Holds the absolute tolerance of a box coordinate, in pixels: 2048 times the float unit
     *        roundoff, the smallest power of two not tighter than the earlier 1e-4f.
     */
    constexpr float K_PIXEL_TOLERANCE =
        static_cast<float>(2048.0 * TestSupport::unit_roundoff<float>());

    /** @brief Holds the seed of the shuffled datasets below. */
    constexpr std::uint64_t K_ORDER_SEED = 42;

    /** @brief Provides a host engine and the builders of the lists these tests read. */
    class ReaderBench : public ::testing::Test
    {
        protected:
            /**
             * @brief Returns a list of 2 by 2 gray samples stored in a file nothing reads.
             * @param count Samples; sample i has the class i modulo 2.
             */
            static SampleList stored_samples(std::size_t count)
            {
                SampleList list;
                list.files.push_back("never_read.bin");
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

            /** @brief Returns stored_samples whose file is written, with rising bytes. */
            static SampleList stored_list(const Support::ScratchDirectory& scratch,
                                          std::size_t count)
            {
                SampleList list = stored_samples(count);
                list.files[0] = scratch.write("pixels.bin", Support::rising_bytes(count * 4, 0));
                return list;
            }

            /** @brief Returns the options of batches of two gray 2 by 2 inputs. */
            static ReaderDatasetOptions gray_options()
            {
                ReaderDatasetOptions options;
                options.fit.size = ImageSize{2, 2};
                options.channels = 1;
                options.batch_size = 2;
                options.name = "tiny";
                return options;
            }

            /** @brief Returns a classification collator over two classes. */
            static std::unique_ptr<const Collator> classification()
            {
                Result<std::unique_ptr<ClassificationCollator>> made =
                    ClassificationCollator::make(2);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }

            /** @brief Makes a dataset of stored samples on the host engine. */
            Result<std::unique_ptr<ReaderDataset>> make(SampleList list,
                                                        const ReaderDatasetOptions& options)
            {
                return ReaderDataset::make(backend.engine(), std::move(list), classification(),
                                           options);
            }

            /** @brief Holds the host engine the batches are allocated from. */
            HostExecutionBackend backend;
    };

    /** @brief Holds the tests that reach no file. */
    class Small_DatasetsReaderDataset : public ReaderBench
    {
    };

    /** @brief Holds the tests that read or write a file. */
    class Medium_DatasetsReaderDataset : public ReaderBench
    {
    };

    /** @brief Holds the options a make refuses and the text of the refusal. */
    struct OptionsRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the width and height of the input. */
        std::int32_t side;

        /** @brief Holds the channels of the input. */
        std::int32_t channels;

        /** @brief Holds the batch size. */
        std::int32_t batch_size;

        /** @brief Holds a text the refusal carries. */
        const char* message;

        /** @brief Holds another text the refusal carries, or none. */
        const char* detail;
    };

    /** @brief Holds one refused option set per row. */
    class Small_DatasetsReaderDatasetOptions
        : public ReaderBench,
          public ::testing::WithParamInterface<OptionsRefusal>
    {
    };

    /** @brief Holds a range and a batch index a read refuses, and the text of the refusal. */
    struct RangeRefusal
    {
        /** @brief Names the row. */
        const char* name;

        /** @brief Holds the index of the batch. */
        std::size_t batch;

        /** @brief Holds the rows asked for. */
        Data::RowRange rows;

        /** @brief Holds a text the refusal carries. */
        const char* message;

        /** @brief Holds another text the refusal carries, or none. */
        const char* detail;
    };

    /** @brief Holds one refused range per row. */
    class Small_DatasetsReaderDatasetRange
        : public ReaderBench,
          public ::testing::WithParamInterface<RangeRefusal>
    {
    };

    /** @brief Holds the epoch of an unshuffled read, one epoch per row. */
    class Medium_DatasetsReaderDatasetEpoch
        : public ReaderBench,
          public ::testing::WithParamInterface<std::int64_t>
    {
    };

    /** @brief Returns the list index of the sample in each row of one batch. */
    std::vector<std::uint64_t> samples_in_batch(const Data::Dataset& dataset, std::size_t index)
    {
        Data::Batch batch;
        const Status status = dataset.read(index, batch);
        EXPECT_TRUE(status.is_ok()) << status.message();

        std::vector<std::uint64_t> samples;
        if (!status.is_ok())
            return samples;

        const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
        for (std::int64_t row = 0; row < dataset.batch_size(); ++row)
            samples.push_back(static_cast<std::uint64_t>(inputs[row * 4] * 255.0 + 0.5) / 4);

        return samples;
    }

    /**
     * @brief Reads every batch and returns the list index of the sample in each row.
     *
     * Sample i of stored_list opens with the byte 4i, which the input holds as 4i / 255.
     */
    std::vector<std::uint64_t> samples_read(const Data::Dataset& dataset)
    {
        std::vector<std::uint64_t> samples;
        for (std::size_t index = 0; index < dataset.batch_count(); ++index)
        {
            const std::vector<std::uint64_t> batch = samples_in_batch(dataset, index);
            samples.insert(samples.end(), batch.begin(), batch.end());
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

    /** @brief Expects an input tensor to hold the bytes first, first + 1, ... divided by 255. */
    void expect_rising_inputs(const SushiBLAS::Tensor& inputs, std::size_t first)
    {
        const std::vector<double> values = SaTest::read_tensor(inputs);
        for (std::size_t i = 0; i < values.size(); ++i)
            EXPECT_NEAR(values[i], static_cast<double>(first + i) / 255.0, K_SCALE_TOLERANCE)
                << i;
    }

    /** @brief Returns a list of two RGB files of 2 by 4, the second holding an object of class 2. */
    SampleList two_detection_files(const Support::ScratchDirectory& scratch)
    {
        SampleList list;
        list.class_count = 3;
        for (const char* name : {"a.saim", "b.saim"})
        {
            Sample sample;
            sample.file = static_cast<std::uint32_t>(list.files.size());
            list.files.push_back(Support::write_raw_image(
                scratch, name, Support::uniform_image(ImageSize{2, 4}, 3, 255)));
            list.samples.push_back(sample);
        }
        list.samples[1].objects.add(2, NormalizedBox{0.5f, 0.5f, 0.5f, 1.0f});
        return list;
    }

    /** @brief Returns the options that letterbox to 4 by 4 in batches of two. */
    ReaderDatasetOptions letterbox_options()
    {
        ReaderDatasetOptions options;
        options.fit.size = ImageSize{4, 4};
        options.fit.fit = SampleFit::Letterbox;
        options.batch_size = 2;
        return options;
    }

    /** @brief Reads the first batch of the detection dataset the tests below share. */
    Data::Batch read_detection_batch(HostExecutionBackend& backend,
                                     const Support::ScratchDirectory& scratch)
    {
        auto collator = DetectionCollator::make(3, 2);
        EXPECT_TRUE(collator.status().is_ok());
        std::unique_ptr<const Collator> owned = std::move(collator).value();
        auto dataset = ReaderDataset::make(backend.engine(), two_detection_files(scratch),
                                           std::move(owned), letterbox_options());
        EXPECT_TRUE(dataset.status().is_ok());

        Data::Batch batch;
        EXPECT_TRUE(SaTest::status_is_ok(std::move(dataset).value()->read(0, batch)));
        return batch;
    }
} // namespace

/** @brief Tests that the dataset reports the batch size, the shapes and the whole batches. */
TEST_F(Small_DatasetsReaderDataset, ItDescribesItsBatches)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_samples(5), gray_options()));

    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->sample_shape(), (Shape{2, 2, 1}));
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->name(), "tiny");
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_LABELS);
}

/** @brief Tests that batch i holds samples 2i and 2i + 1, in the order of the list. */
TEST_F(Medium_DatasetsReaderDataset, ABatchHoldsItsSamplesInListOrder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_order");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 4), gray_options()));

    EXPECT_EQ(samples_in_batch(*dataset, 1), (std::vector<std::uint64_t>{2, 3}));
}

/** @brief Tests that the inputs of a batch are the bytes of its samples divided by 255. */
TEST_F(Medium_DatasetsReaderDataset, TheInputsOfABatchAreScaledBy255)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_scaled");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 4), gray_options()));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    expect_rising_inputs(batch.inputs, 8);
}

/** @brief Tests that the labels of a batch follow the classes of the list. */
TEST_F(Medium_DatasetsReaderDataset, TheLabelsOfABatchFollowTheList)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_labels");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 4), gray_options()));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    ASSERT_EQ(batch.targets.size(), 1u);
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values), (std::vector<double>{0.0, 1.0}));
}

/** @brief Tests that a detection batch letterboxes each file and pads the rows above it. */
TEST_F(Medium_DatasetsReaderDataset, ADetectionBatchLetterboxesTheImages)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_detection_image");

    const Data::Batch batch = read_detection_batch(backend, scratch);

    // Row 0 of each 4 by 4 by 3 input is padding of 114; row 1 is the image.
    const std::vector<double> inputs = SaTest::read_tensor(batch.inputs);
    EXPECT_NEAR(inputs[0], 114.0 / 255.0, K_SCALE_TOLERANCE);
    EXPECT_NEAR(inputs[12], 1.0, K_SCALE_TOLERANCE);
}

/** @brief Tests that the boxes of a detection batch move with the letterboxed image. */
TEST_F(Medium_DatasetsReaderDataset, ADetectionBatchMovesTheBoxesWithTheImage)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_detection_boxes");

    const Data::Batch batch = read_detection_batch(backend, scratch);

    ASSERT_EQ(batch.targets.size(), 3u);
    const std::vector<double> boxes = SaTest::read_tensor(batch.targets[0].values);
    EXPECT_NEAR(boxes[8], 1.0, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[9], 1.0, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[10], 3.0, K_PIXEL_TOLERANCE);
    EXPECT_NEAR(boxes[11], 3.0, K_PIXEL_TOLERANCE);
}

/** @brief Tests that a detection batch flags the one object row and names its class. */
TEST_F(Medium_DatasetsReaderDataset, ADetectionBatchFlagsTheObjectRowAndItsClass)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_detection_valid");

    const Data::Batch batch = read_detection_batch(backend, scratch);

    ASSERT_EQ(batch.targets.size(), 3u);
    EXPECT_EQ(SaTest::read_tensor(batch.targets[2].values),
              (std::vector<double>{0.0, 0.0, 1.0, 0.0}));
    EXPECT_EQ(SaTest::read_tensor(batch.targets[1].values)[2], 2.0);
}

/** @brief Tests that make refuses options no batch can be built from. */
TEST_P(Small_DatasetsReaderDatasetOptions, MakeRefusesOptionsOutsideTheirBounds)
{
    const OptionsRefusal& row = GetParam();
    ReaderDatasetOptions options = gray_options();
    options.fit.size = ImageSize{row.side, 2};
    options.channels = row.channels;
    options.batch_size = row.batch_size;

    SA_EXPECT_FAILURE(ReaderDataset::make(backend.engine(), stored_samples(2), classification(),
                                          options),
                      StatusCode::InvalidDescription, row.message, row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsReaderDatasetOptions,
    ::testing::Values(OptionsRefusal{"ASizeOfNoPixel", 0, 1, 2, "0 by 2", ""},
                      OptionsRefusal{"FiveChannels", 2, 5, 2, "5 channels", ""},
                      OptionsRefusal{"ABatchSizeOfZero", 2, 1, 0, "batch size of 0", ""},
                      OptionsRefusal{"ABatchLargerThanTheList", 2, 1, 3, "2 samples",
                                     "one batch of 3"}),
    [](const ::testing::TestParamInfo<OptionsRefusal>& info) { return std::string(info.param.name); });

/** @brief Tests that make refuses a missing collator. */
TEST_F(Small_DatasetsReaderDataset, MakeRefusesANullCollator)
{
    SA_EXPECT_FAILURE(ReaderDataset::make(backend.engine(), stored_samples(2), nullptr,
                                          gray_options()),
                      StatusCode::InvalidDescription, "ReaderDataset", "tiny", "no collator");
}

/** @brief Tests that a batch index past the last batch stops the process. */
TEST_F(Medium_DatasetsReaderDataset, ABatchOutsideTheEpochStopsTheProcess)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_samples(2), gray_options()));

    Data::Batch batch;
    SA_EXPECT_FATAL((void)dataset->read(1, batch), "ReaderDataset", "1 batches", "batch 1");
}

/** @brief Tests that a second read into the same batch holds the values of the second batch. */
TEST_F(Medium_DatasetsReaderDataset, AReadIntoAFilledBatchHoldsTheValuesOfTheSecondRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_reuse");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 4), gray_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    expect_rising_inputs(batch.inputs, 8);
    EXPECT_EQ(SaTest::read_tensor(batch.targets[0].values), (std::vector<double>{0.0, 1.0}));
}

/** @brief Tests that a read leaves alone a tensor something else still references. */
TEST_F(Medium_DatasetsReaderDataset, AReadDoesNotWriteATensorSomethingElseHolds)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_shared");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 4), gray_options()));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const SushiBLAS::Tensor kept = batch.inputs;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, batch)));

    EXPECT_NE(batch.inputs.data_as<float>(), kept.data_as<float>());
    EXPECT_NEAR(kept.data_as<float>()[0], 0.0f, K_SCALE_TOLERANCE);
    EXPECT_NEAR(batch.inputs.data_as<float>()[0], 8.0f / 255.0f, K_SCALE_TOLERANCE);
}

/** @brief Tests that a reader dataset is read through the Data::Dataset interface. */
TEST_F(Medium_DatasetsReaderDataset, ItIsADatasetATrainerCanRead)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_seam");
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> made,
                            make(stored_list(scratch, 4), gray_options()));
    const Data::Dataset& dataset = *made;

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, batch)));
    ASSERT_EQ(batch.targets.size(), dataset.targets().size());
    EXPECT_EQ(batch.targets[0].role, dataset.targets()[0].role);
    EXPECT_EQ(batch.inputs.shape[0], dataset.batch_size());
}

/** @brief Tests that an image of another channel count is a failure naming its file. */
TEST_F(Medium_DatasetsReaderDataset, AnImageOfAnotherChannelCountNamesItsFile)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_channels");
    ReaderDatasetOptions options = gray_options();
    options.channels = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 2), options));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::Incompatible, "pixels.bin",
                      "ReaderDataset", "1 channels", "reads 3");
}

/** @brief Tests that a file that is absent when its batch is read is a failure naming it. */
TEST_F(Medium_DatasetsReaderDataset, AFileAbsentAtReadTimeIsNamed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_absent");
    SampleList list = stored_list(scratch, 2);
    list.files[0] = scratch.path() / "gone.bin";
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(std::move(list), gray_options()));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::InvalidFile, "gone.bin");
}

/** @brief Tests that a shuffled epoch reads every sample once. */
TEST_F(Medium_DatasetsReaderDataset, AShuffledEpochReadsEverySampleOnce)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_shuffle_once");
    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 8), options));

    const std::vector<std::uint64_t> read = samples_read(*dataset);

    std::vector<std::uint64_t> every_sample(8);
    std::iota(every_sample.begin(), every_sample.end(), std::uint64_t{0});
    EXPECT_TRUE(std::is_permutation(read.begin(), read.end(), every_sample.begin(),
                                    every_sample.end()));
}

/** @brief Tests that a shuffled epoch reads the samples in the order of its seed. */
TEST_F(Medium_DatasetsReaderDataset, AShuffledEpochFollowsTheOrderOfTheSeed)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_shuffle_order");
    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 8), options));

    EXPECT_EQ(samples_read(*dataset), order_of(K_ORDER_SEED, 0, 8));
}

/** @brief Tests that begin_epoch gives the order of that epoch. */
TEST_F(Medium_DatasetsReaderDataset, BeginEpochSelectsTheOrderOfTheEpoch)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_epochs");
    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 8), options));

    Data::begin_epoch(*dataset, 3);

    const std::vector<std::uint64_t> third = samples_read(*dataset);
    EXPECT_EQ(third, order_of(K_ORDER_SEED, 3, 8));
    EXPECT_NE(third, order_of(K_ORDER_SEED, 0, 8));
}

/** @brief Tests that beginning the first epoch again gives the first order again. */
TEST_F(Medium_DatasetsReaderDataset, BeginningTheFirstEpochAgainRestoresItsOrder)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_epochs_again");
    ReaderDatasetOptions options = gray_options();
    options.shuffle = true;
    options.seed = K_ORDER_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 8), options));
    Data::begin_epoch(*dataset, 3);

    Data::begin_epoch(*dataset, 0);

    EXPECT_EQ(samples_read(*dataset), order_of(K_ORDER_SEED, 0, 8));
}

/** @brief Tests that without shuffle every epoch reads the samples in the list's order. */
TEST_P(Medium_DatasetsReaderDatasetEpoch, WithoutShuffleEveryEpochKeepsTheOrderOfTheList)
{
    const Support::ScratchDirectory scratch(
        ("sushiai_test_datasets_reader_unshuffled_" + std::to_string(GetParam())).c_str());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 6), gray_options()));

    Data::begin_epoch(*dataset, GetParam());

    EXPECT_EQ(samples_read(*dataset), (std::vector<std::uint64_t>{0, 1, 2, 3, 4, 5}));
}

INSTANTIATE_TEST_SUITE_P(Rows, Medium_DatasetsReaderDatasetEpoch,
                         ::testing::Values<std::int64_t>(0, 5),
                         [](const ::testing::TestParamInfo<std::int64_t>& info)
                         {
                             return "Epoch" + std::to_string(info.param);
                         });

/** @brief Tests that the form of make that takes a pipeline refuses a null one. */
TEST_F(Small_DatasetsReaderDataset, MakeRefusesANullPipeline)
{
    SA_EXPECT_FAILURE(ReaderDataset::make(backend.engine(), stored_samples(2), classification(),
                                          nullptr, gray_options()),
                      StatusCode::InvalidDescription, "ReaderDataset", "tiny", "no pipeline");
}

/** @brief Tests that a range read of a shuffled epoch holds the rows the seed order names. */
TEST_F(Medium_DatasetsReaderDataset, ARangeReadHoldsTheRowsTheSeedOrderNames)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_rows");
    ReaderDatasetOptions options = gray_options();
    options.batch_size = 4;
    options.shuffle = true;
    options.seed = K_ORDER_SEED;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_list(scratch, 8), options));
    dataset->begin_epoch(1);

    Data::Batch part;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(1, Data::RowRange{1, 2}, part)));

    // Rows 1 and 2 of batch 1 are the samples at positions 5 and 6 of the epoch order.
    const std::vector<std::uint64_t> order = order_of(K_ORDER_SEED, 1, 8);
    const std::vector<double> inputs = SaTest::read_tensor(part.inputs);
    ASSERT_EQ(inputs.size(), 8u);
    for (std::size_t row = 0; row < 2; ++row)
    {
        for (std::size_t pixel = 0; pixel < 4; ++pixel)
            EXPECT_NEAR(inputs[row * 4 + pixel],
                        static_cast<double>(order[5 + row] * 4 + pixel) / 255.0,
                        K_SCALE_TOLERANCE)
                << "row " << row << " pixel " << pixel;
    }
}

/** @brief Tests that a range read opens no file of a sample outside the range. */
TEST_F(Medium_DatasetsReaderDataset, ARangeReadOpensNoSampleOutsideIt)
{
    const Support::ScratchDirectory scratch("sushiai_test_datasets_reader_rows_absent");
    SampleList list = stored_list(scratch, 2);
    list.files.push_back(scratch.path() / "gone.bin");
    list.samples[0].file = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(std::move(list), gray_options()));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(0, Data::RowRange{1, 1}, batch)));
    EXPECT_EQ(batch.inputs.shape[0], 1);
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::InvalidFile, "gone.bin");
}

/** @brief Tests that a range outside the reader dataset is refused with a Status, not a stop. */
TEST_P(Small_DatasetsReaderDatasetRange, RegressionABadRangeIsRefusedWithAStatus)
{
    const RangeRefusal& row = GetParam();
    ReaderDatasetOptions options = gray_options();
    options.batch_size = 4;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReaderDataset> dataset,
                            make(stored_samples(8), options));
    Data::Batch batch;

    SA_EXPECT_FAILURE(dataset->read_rows(row.batch, row.rows, batch),
                      StatusCode::InvalidDescription, row.message, row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_DatasetsReaderDatasetRange,
    ::testing::Values(RangeRefusal{"RowsPastTheBatch", 1, Data::RowRange{3, 2},
                                   "Rows 3 to 5", "of batch 1"},
                      RangeRefusal{"ABatchPastTheEpoch", 99, Data::RowRange{0, 1},
                                   "Batch 99 is out of range", ""}),
    [](const ::testing::TestParamInfo<RangeRefusal>& info) { return std::string(info.param.name); });
