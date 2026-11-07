/**************************************************************************/
/* test_dataset.cpp                                                       */
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
 * @file test_dataset.cpp
 * @brief Tests what the generated datasets state, how every dataset reads a batch or a range of
 *        its rows, and what the dataset factories refuse.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>

#include "batch_rows.hpp"

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples per batch of every dataset the contract suites build. */
    constexpr std::int64_t K_CONTRACT_ROWS = 4;

    /** @brief Returns the options of a small synthetic set: 2 batches of 4 samples, 3 features. */
    Data::SyntheticOptions small_synthetic()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_CONTRACT_ROWS;
        options.batches = 2;
        options.features = 3;
        options.classes = 2;
        return options;
    }

    /** @brief Returns the options of a small rings set: 3 batches of 4 samples. */
    Data::RingsOptions small_rings()
    {
        Data::RingsOptions options;
        options.batch_size = K_CONTRACT_ROWS;
        options.batches = 3;
        return options;
    }

    /** @brief Returns the value of @p made as a Dataset, or null after a failure it reports. */
    template <typename Made>
    std::unique_ptr<Data::Dataset> held(Made&& made)
    {
        EXPECT_TRUE(SaTest::status_is_ok(Detail::status_of(made)));
        if (!made.has_value())
            return nullptr;

        return std::move(made).value();
    }

    /** @brief Builds a synthetic set on @p backend. */
    std::unique_ptr<Data::Dataset> synthetic_on(SushiAI::ExecutionBackend& backend)
    {
        return held(Data::SyntheticClassification::make(backend, small_synthetic()));
    }

    /** @brief Builds a rings set on @p backend. */
    std::unique_ptr<Data::Dataset> rings_on(SushiAI::ExecutionBackend& backend)
    {
        return held(Data::Rings::make(backend.engine(), small_rings()));
    }

    /** @brief Builds a materialised set of two batches of 4 samples on @p backend. */
    std::unique_ptr<Data::Dataset> materialised_on(SushiAI::ExecutionBackend& backend)
    {
        const Data::InputDescription input{Shape{4}, DType::Float32};
        const std::vector<Data::TargetDescription> roles{
            Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32, 2}};
        std::vector<Data::Batch> batches;
        for (int index = 0; index < 2; ++index)
        {
            Result<Data::Batch> batch =
                Data::allocate_batch(backend.engine(), K_CONTRACT_ROWS, input, roles);
            EXPECT_TRUE(SaTest::status_is_ok(Detail::status_of(batch)));
            if (!batch.has_value())
                return nullptr;
            batches.push_back(std::move(batch).value());
        }

        return std::make_unique<Data::MaterialisedDataset>(std::move(batches), K_CONTRACT_ROWS,
                                                           input, roles, "held");
    }

    /** @brief Builds a threaded loader over a rings set on @p backend. */
    std::unique_ptr<Data::Dataset> loader_on(SushiAI::ExecutionBackend& backend)
    {
        std::unique_ptr<Data::Dataset> rings = rings_on(backend);
        if (rings == nullptr)
            return nullptr;

        return held(Data::ThreadedLoader::make(std::move(rings), Data::LoaderOptions{}));
    }

    /** @brief Names one kind of dataset and how to build it. */
    struct DatasetKind
    {
        /** @brief Names the kind in the test report. */
        const char* name;

        /** @brief Builds a dataset of the kind with batches of K_CONTRACT_ROWS samples. */
        std::unique_ptr<Data::Dataset> (*make)(SushiAI::ExecutionBackend& backend);
    };

    /** @brief Returns the name of the kind a parameter holds, for the test report. */
    std::string kind_name(const ::testing::TestParamInfo<DatasetKind>& info)
    {
        return info.param.name;
    }

    /** @brief Names a range of rows and the batch it is read from. */
    struct RangeCase
    {
        /** @brief Counts the batch in the dataset. */
        std::size_t batch;

        /** @brief Names the rows read. */
        Data::RowRange rows;
    };

    /** @brief Provides the engine the datasets are generated on. */
    class Medium_Dataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the inputs of batch @p index of a synthetic set seeded @p seed. */
            std::vector<float> synthetic_inputs(std::uint64_t seed, std::size_t index)
            {
                Data::SyntheticOptions options = small_synthetic();
                options.seed = seed;
                std::vector<float> values;
                Result<std::unique_ptr<Data::SyntheticClassification>> made =
                    Data::SyntheticClassification::make(backend(), options);
                EXPECT_TRUE(SaTest::status_is_ok(Detail::status_of(made)));
                if (!made.has_value())
                    return values;

                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(made.value()->read(index, batch)));
                const float* const data = batch.inputs.data_as<float>();
                values.assign(data, data + batch.inputs.num_elements);
                return values;
            }
    };

    /** @brief Runs each case on a kind of dataset whose batches the dataset holds. */
    class Medium_DatasetSharing : public SaTest::EngineTest,
                                  public ::testing::WithParamInterface<DatasetKind>
    {

    };

    /** @brief Runs each case on a kind of dataset and a range of rows. */
    class Medium_DatasetRange
        : public SaTest::EngineTest,
          public ::testing::WithParamInterface<std::tuple<DatasetKind, RangeCase>>
    {

    };

    /** @brief Runs each case on a kind of dataset. */
    class Medium_DatasetRefusal : public SaTest::EngineTest,
                                  public ::testing::WithParamInterface<DatasetKind>
    {

    };

    /** @brief Names one factory call with a non-positive extent and what its refusal names. */
    struct ExtentCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Calls the factory and returns the Status it gave. */
        std::function<Status(SushiAI::ExecutionBackend& backend)> build;

        /** @brief Holds a text the refusal names besides "positive". */
        const char* detail;
    };

    /** @brief Builds a synthetic set with no feature and returns the Status of that. */
    Status synthetic_without_features(SushiAI::ExecutionBackend& backend)
    {
        Data::SyntheticOptions options = small_synthetic();
        options.features = 0;
        return Detail::status_of(Data::SyntheticClassification::make(backend, options));
    }

    /** @brief Builds a rings set through the registry with a batch size of 0 and returns the Status. */
    Status registered_rings_without_rows(SushiAI::ExecutionBackend& backend)
    {
        Data::DatasetRegistry datasets;
        const Status registered = Data::register_datasets(datasets);
        if (!registered.is_ok())
            return registered;

        return Detail::status_of(Data::make_dataset(
            datasets, backend.engine(), "rings",
            SaTest::attributes_from(R"({"batch_size": 0})"), "data"));
    }

    /** @brief Runs each case on the engine. */
    class Medium_DatasetExtent : public SaTest::EngineTest,
                                 public ::testing::WithParamInterface<ExtentCase>
    {

    };

    /** @brief Names one option of Rings that is out of range and what its refusal names. */
    struct OptionCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the options with one value out of range. */
        Data::RingsOptions options;

        /** @brief Holds a text the refusal names. */
        const char* detail;
    };

    /** @brief Returns the rings options with one value out of range, per row of the table. */
    std::vector<OptionCase> option_cases()
    {
        Data::RingsOptions distractors;
        distractors.distractor_features = -1;
        Data::RingsOptions width;
        width.ring_width = 0.0;
        Data::RingsOptions noise;
        noise.position_noise = -0.5;

        return {OptionCase{"ANegativeDistractorCount", distractors, "distractor"},
                OptionCase{"AZeroRingWidth", width, "ring width"},
                OptionCase{"ANegativePositionNoise", noise, "position noise"}};
    }

    /** @brief Runs each case on the engine. */
    class Medium_RingsOption : public SaTest::EngineTest,
                               public ::testing::WithParamInterface<OptionCase>
    {

    };
} // namespace

/** @brief Tests that a read of a synthetic set gives inputs of batch size by feature count. */
TEST_F(Medium_Dataset, AReadGivesInputsOfBatchSizeByFeatureCount)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(backend(), small_synthetic()));
    Data::Batch batch;

    SA_EXPECT_OK(dataset->read(1, batch));

    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], K_CONTRACT_ROWS);
    EXPECT_EQ(batch.inputs.shape[1], 3);
}

/** @brief Tests that one seed gives the same inputs and another seed other inputs. */
TEST_F(Medium_Dataset, TheSeedFixesTheSyntheticInputs)
{
    const std::vector<float> first = synthetic_inputs(5, 1);

    EXPECT_EQ(first, synthetic_inputs(5, 1));
    EXPECT_NE(first, synthetic_inputs(6, 1));
}

/** @brief Tests that a factory refuses a non-positive extent and names it. */
TEST_P(Medium_DatasetExtent, AFactoryRefusesANonPositiveExtent)
{
    const ExtentCase& row = GetParam();

    SA_EXPECT_FAILURE(row.build(backend()), StatusCode::InvalidDescription, "positive", row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Factories, Medium_DatasetExtent,
    ::testing::Values(ExtentCase{"SyntheticClassificationWithNoFeature", &synthetic_without_features,
                                 "features=0"},
                      ExtentCase{"RingsFromTheRegistryWithNoRow", &registered_rings_without_rows,
                                 "positive"}),
    [](const ::testing::TestParamInfo<ExtentCase>& info)
    {
        return std::string(info.param.name);
    });

/** @brief Tests that Rings refuses an option out of range and names it. */
TEST_P(Medium_RingsOption, RingsRefusesAnOptionOutOfRange)
{
    const OptionCase& row = GetParam();

    SA_EXPECT_FAILURE(Data::Rings::make(*engine, row.options), StatusCode::InvalidDescription,
                      row.detail);
}

INSTANTIATE_TEST_SUITE_P(Options, Medium_RingsOption, ::testing::ValuesIn(option_cases()),
                         [](const ::testing::TestParamInfo<OptionCase>& info)
                         {
                             return std::string(info.param.name);
                         });

/** @brief Tests that two reads of one batch share its tensors and a read of another does not. */
TEST_P(Medium_DatasetSharing, AReadSharesTheHeldBatch)
{
    const std::unique_ptr<Data::Dataset> dataset = GetParam().make(backend());
    ASSERT_NE(dataset, nullptr);
    Data::Batch first;
    Data::Batch again;
    Data::Batch other;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, first)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, again)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, other)));

    EXPECT_EQ(first.inputs.data(), again.inputs.data());
    ASSERT_EQ(first.targets.size(), 1u);
    ASSERT_EQ(again.targets.size(), 1u);
    EXPECT_EQ(first.targets[0].values.data(), again.targets[0].values.data());
    EXPECT_NE(first.inputs.data(), other.inputs.data());
}

/** @brief Tests that a range read views the storage of the held batch and copies no element. */
TEST_P(Medium_DatasetSharing, ARangeReadSharesTheStorageOfTheHeldBatch)
{
    const std::unique_ptr<Data::Dataset> dataset = GetParam().make(backend());
    ASSERT_NE(dataset, nullptr);
    Data::Batch whole;
    Data::Batch part;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(1, whole)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(1, Data::RowRange{1, 2}, part)));

    EXPECT_EQ(part.inputs.storage.get(), whole.inputs.storage.get());
}

INSTANTIATE_TEST_SUITE_P(HoldingKinds, Medium_DatasetSharing,
                         ::testing::Values(DatasetKind{"SyntheticClassification", &synthetic_on},
                                           DatasetKind{"Rings", &rings_on},
                                           DatasetKind{"MaterialisedDataset", &materialised_on}),
                         kind_name);

/** @brief Tests that a range read gives the rows the whole read holds, byte for byte. */
TEST_P(Medium_DatasetRange, ARangeReadGivesTheRowsOfTheWholeBatch)
{
    const DatasetKind& kind = std::get<0>(GetParam());
    const RangeCase& range = std::get<1>(GetParam());
    const std::unique_ptr<Data::Dataset> dataset = kind.make(backend());
    ASSERT_NE(dataset, nullptr);
    Data::Batch whole;
    Data::Batch part;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(range.batch, whole)));
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read_rows(range.batch, range.rows, part)));

    EXPECT_EQ(part.inputs.shape[0], range.rows.count);
    EXPECT_TRUE(DataTest::holds_rows_of(whole, dataset->batch_size(), range.rows, part));
}

INSTANTIATE_TEST_SUITE_P(
    Ranges, Medium_DatasetRange,
    ::testing::Combine(::testing::Values(DatasetKind{"SyntheticClassification", &synthetic_on},
                                         DatasetKind{"Rings", &rings_on},
                                         DatasetKind{"MaterialisedDataset", &materialised_on},
                                         DatasetKind{"ThreadedLoader", &loader_on}),
                       ::testing::Values(RangeCase{1, Data::RowRange{1, 2}},
                                         RangeCase{0, Data::RowRange{3, 1}})),
    [](const ::testing::TestParamInfo<std::tuple<DatasetKind, RangeCase>>& info)
    {
        return std::string(std::get<0>(info.param).name) + "Batch"
               + std::to_string(std::get<1>(info.param).batch);
    });

/** @brief Tests that a range outside a dataset is refused with a Status, not a stop. */
TEST_P(Medium_DatasetRefusal, ABadRangeIsRefusedWithAStatus)
{
    const std::unique_ptr<Data::Dataset> dataset = GetParam().make(backend());
    ASSERT_NE(dataset, nullptr);
    const std::int64_t size = dataset->batch_size();
    Data::Batch batch;

    SA_EXPECT_FAILURE((dataset->read_rows(1, Data::RowRange{size - 1, 2}, batch)),
                      StatusCode::InvalidDescription,
                      "Rows " + std::to_string(size - 1) + " to " + std::to_string(size + 1),
                      "of batch 1", "a batch of " + std::to_string(size));
    SA_EXPECT_FAILURE((dataset->read_rows(99, Data::RowRange{0, 1}, batch)),
                      StatusCode::InvalidDescription, "Batch 99 is out of range");
}

INSTANTIATE_TEST_SUITE_P(Kinds, Medium_DatasetRefusal,
                         ::testing::Values(DatasetKind{"SyntheticClassification", &synthetic_on},
                                           DatasetKind{"Rings", &rings_on},
                                           DatasetKind{"MaterialisedDataset", &materialised_on},
                                           DatasetKind{"ThreadedLoader", &loader_on}),
                         kind_name);

namespace
{
    /** @brief Names a run of labels: where it starts, the classes it spans and what it holds. */
    struct LabelRun
    {
        /** @brief Holds the position, over the whole set, of the first label. */
        std::int64_t first;

        /** @brief Holds how many classes the labels span. */
        std::int64_t classes;

        /** @brief Holds the label of every sample of the run. */
        std::vector<std::int32_t> labels;
    };

    /** @brief Runs each case with a run of labels. */
    class Small_SyntheticLabels : public ::testing::TestWithParam<LabelRun>
    {

    };
} // namespace

/** @brief Tests that the labels go round the classes from the position the run starts at. */
TEST_P(Small_SyntheticLabels, LabelsRunRoundRobinOverTheWholeSet)
{
    const LabelRun& run = GetParam();
    std::vector<std::int32_t> labels(run.labels.size());

    Data::fill_round_robin_labels(run.first, run.classes,
                                  SushiRuntime::span<std::int32_t>(labels.data(), labels.size()));

    EXPECT_EQ(labels, run.labels);
}

INSTANTIATE_TEST_SUITE_P(Runs, Small_SyntheticLabels,
                         ::testing::Values(LabelRun{0, 3, {0, 1, 2, 0, 1, 2, 0}},
                                           LabelRun{5, 3, {2, 0, 1, 2}},
                                           LabelRun{0, 1, {0, 0, 0}}));

/** @brief Tests that each row takes the mean row of its class. */
TEST(Small_SyntheticMeans, EachRowTakesTheMeanOfItsClass)
{
    const std::vector<float> means = {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f};
    const std::vector<std::int32_t> labels = {2, 0, 2, 1};
    std::vector<float> rows(labels.size() * 2, -1.0f);

    Data::copy_class_means(SushiRuntime::span<const float>(means.data(), means.size()), 2,
                           SushiRuntime::span<const std::int32_t>(labels.data(), labels.size()),
                           SushiRuntime::span<float>(rows.data(), rows.size()));

    EXPECT_EQ(rows, (std::vector<float>{100.0f, 200.0f, 1.0f, 2.0f, 100.0f, 200.0f, 10.0f, 20.0f}));
}
