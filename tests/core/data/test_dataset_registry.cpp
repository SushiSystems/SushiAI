/**************************************************************************/
/* test_dataset_registry.cpp                                              */
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
 * @file test_dataset_registry.cpp
 * @brief Tests the dataset registry: its entries, the roles and targets they declare, and what it
 *        refuses.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /**
     * @brief Bounds the absolute error of a float32 point's radius; 2048 unit roundoffs of float, the
     * smallest power of two that covers 1e-4.
     */
    constexpr double K_RADIUS_TOLERANCE = 2048.0 * TestSupport::unit_roundoff<float>();

    /** @brief The "rings" row, as register_datasets adds it. */
    constexpr Data::DatasetEntry K_RINGS_ROW{"rings", &Data::make_rings};

    /** @brief A second row, under a type no module registers. */
    constexpr Data::DatasetEntry K_SECOND_ROW{"moons", &Data::make_rings};
} // namespace

/** @brief Tests that rows() returns the rows in the order add() received them. */
TEST(Small_DatasetRegistry, AddKeepsInsertionOrder)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(registry.add(K_RINGS_ROW));
    SA_EXPECT_OK(registry.add(K_SECOND_ROW));

    const SushiRuntime::span<const Data::DatasetEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].type, "rings");
    EXPECT_EQ(rows[1].type, "moons");
}

/** @brief Tests that find returns null for a type no row holds and the row of one it holds. */
TEST(Small_DatasetRegistry, FindReturnsNullForAnUnknownType)
{
    Data::DatasetRegistry registry;
    EXPECT_EQ(registry.find("rings"), nullptr);

    SA_EXPECT_OK(registry.add(K_RINGS_ROW));
    EXPECT_EQ(registry.find("coco"), nullptr);

    const Data::DatasetEntry* row = registry.find("rings");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->type, "rings");
}

/** @brief Tests that add refuses a type a row already holds and names the type. */
TEST(Small_DatasetRegistry, ATypeAddedTwiceIsRefused)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(registry.add(K_RINGS_ROW));

    SA_EXPECT_FAILURE(registry.add(K_RINGS_ROW), StatusCode::InvalidDescription, "'rings'",
                      "already registered");
}

/** @brief Tests that register_datasets adds the "rings" row. */
TEST(Small_DatasetRegistry, RegisterDatasetsAddsTheRingsRow)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Data::register_datasets(registry));

    const Data::DatasetEntry* row = registry.find("rings");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->type, "rings");
}

namespace
{
    class Medium_Registries : public SaTest::EngineTest
    {
        protected:
            /** @brief Adds the "rings" row the dataset tests name. */
            void SetUp() override
            {
                SaTest::EngineTest::SetUp();
                SA_EXPECT_OK(Data::register_datasets(datasets));
            }

            /** @brief Holds the dataset rows the tests build from. */
            Data::DatasetRegistry datasets;
    };

    using Test::attributes_from;

    /** @brief Returns the class index of every sample of @p batch, or nothing. */
    std::vector<int32_t> labels_of(const Data::Batch& batch)
    {
        const Data::BatchTarget* labels = Data::find_target(batch.targets, Data::K_ROLE_LABELS);
        if (labels == nullptr || labels->values.dtype != DType::Int32)
            return {};

        const int32_t* values = labels->values.data_as<int32_t>();
        return std::vector<int32_t>(values, values + labels->values.num_elements);
    }

    /** @brief Returns how many times each class index below @p classes occurs in @p labels. */
    std::vector<int> label_counts(const std::vector<int32_t>& labels, std::size_t classes)
    {
        std::vector<int> counts(classes, 0);
        for (int32_t label : labels)
            ++counts[static_cast<std::size_t>(label)];

        return counts;
    }

    /**
     * @brief Returns whether @p batch holds a row of class @p band and every such row lies in the
     *        radius band [band, band + width).
     */
    ::testing::AssertionResult is_in_radius_band(const Data::Batch& batch, int64_t features,
                                                 std::int32_t band, double width)
    {
        const float* const x = batch.inputs.data_as<float>();
        const std::vector<int32_t> labels = labels_of(batch);

        std::size_t rows_in_band = 0;
        for (std::size_t r = 0; r < labels.size(); ++r)
        {
            if (labels[r] != band)
                continue;

            ++rows_in_band;
            const double px = x[static_cast<int64_t>(r) * features];
            const double py = x[static_cast<int64_t>(r) * features + 1];
            const double radius = std::sqrt(px * px + py * py);
            if (radius < band - K_RADIUS_TOLERANCE || radius >= band + width + K_RADIUS_TOLERANCE)
                return ::testing::AssertionFailure()
                       << "row " << r << " has radius " << radius << " outside the band " << band;
        }

        if (rows_in_band == 0)
            return ::testing::AssertionFailure() << "no row of class " << band;

        return ::testing::AssertionSuccess();
    }

    /** @brief Runs each case on the radius band of one class. */
    class Medium_RingsBand : public Medium_Registries,
                             public ::testing::WithParamInterface<std::int32_t>
    {

    };

    /** @brief Names a key the AdamW factory refuses and what its refusal names. */
    struct UnknownKeyCase
    {
        /** @brief Names the case in the test report. */
        const char* name;

        /** @brief Holds the attributes that carry the key. */
        const char* attributes;

        /** @brief Holds the key as the refusal quotes it. */
        const char* quoted_key;

        /** @brief Holds a second text the refusal names. */
        const char* detail;
    };

    /** @brief Runs each case on the engine. */
    class Medium_OptimizerKeys : public Medium_Registries,
                                 public ::testing::WithParamInterface<UnknownKeyCase>
    {

    };
} // namespace

/** @brief Tests that Rings labels the samples round-robin, so every batch holds each class alike. */
TEST_F(Medium_Registries, RingsLabelsEveryBatchBalanced)
{
    Data::RingsOptions options;
    options.batch_size = 32;
    options.batches = 4;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    EXPECT_EQ(label_counts(labels_of(batch), 4), (std::vector<int>{8, 8, 8, 8}));
}

/** @brief Tests that Rings publishes the class index of each sample in rotation. */
TEST_F(Medium_Registries, RingsPublishesItsLabelsAsOneIndexPerSample)
{
    Data::RingsOptions options;
    options.batch_size = 8;
    options.batches = 1;
    options.classes = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    EXPECT_EQ(labels_of(batch), (std::vector<int32_t>{0, 1, 2, 0, 1, 2, 0, 1}));
}

/** @brief Tests that the inputs of a Rings batch carry the batch axis before the sample shape. */
TEST_F(Medium_Registries, RingsInputsCarryTheBatchAxisBeforeTheSampleShape)
{
    Data::RingsOptions options;
    options.batch_size = 8;
    options.batches = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));
    Data::Batch batch;

    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], 8);
    EXPECT_EQ(batch.inputs.shape[1], 2 + options.distractor_features);
}

/**
 * @brief Tests that every sample of a class lies in the radius band of that class.
 *
 * The class is a function of x^2 + y^2, which is degree 2 in the inputs and not a half-space.
 */
TEST_P(Medium_RingsBand, RingsPlacesEachClassInItsOwnRadiusBand)
{
    Data::RingsOptions options;
    options.batch_size = 64;
    options.batches = 1;
    options.position_noise = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));
    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));

    EXPECT_TRUE(is_in_radius_band(batch, dataset->sample_shape().num_elements(), GetParam(),
                                  options.ring_width));
}

INSTANTIATE_TEST_SUITE_P(Classes, Medium_RingsBand, ::testing::Values(0, 1, 2, 3));

/** @brief Tests that the keys of a configuration value reach the options of a rings set. */
TEST(Small_RingsOptions, TheKeysOfAConfigValueReachTheOptions)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::RingsOptions options,
        Data::read_rings_options(attributes_from(R"({"batch_size": 16, "batches": 2,
                                  "classes": 3, "distractors": 1, "ring_width": 0.5,
                                  "position_noise": 0.25, "seed": 7})"),
                                 "data"));

    EXPECT_EQ(options.batch_size, 16);
    EXPECT_EQ(options.batches, 2);
    EXPECT_EQ(options.classes, 3);
    EXPECT_EQ(options.distractor_features, 1);
    EXPECT_EQ(options.ring_width, 0.5);
    EXPECT_EQ(options.position_noise, 0.25);
    EXPECT_EQ(options.seed, 7u);
}

/** @brief Tests that a key a configuration value leaves out keeps the default of its option. */
TEST(Small_RingsOptions, AKeyLeftOutKeepsItsDefault)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::RingsOptions options,
                            Data::read_rings_options(attributes_from("{}"), "data"));
    const Data::RingsOptions defaults;

    EXPECT_EQ(options.batch_size, defaults.batch_size);
    EXPECT_EQ(options.distractor_features, defaults.distractor_features);
    EXPECT_EQ(options.seed, defaults.seed);
}

/** @brief Tests that a value of the wrong type is refused and names its key. */
TEST(Small_RingsOptions, AValueOfTheWrongTypeIsRefused)
{
    SA_EXPECT_FAILURE(Data::read_rings_options(attributes_from(R"({"classes": "three"})"), "data"),
                      StatusCode::InvalidDescription, "data", "classes");
}

/** @brief Tests that two datasets built from one seed hold the same bytes. */
TEST_F(Medium_Registries, TheSameSeedGivesTheSameData)
{
    const auto build = [this]
    {
        return Data::make_dataset(datasets, *engine, "rings",
                                  attributes_from(R"({"batch_size": 8, "batches": 1, "seed": 42})"),
                                  "data");
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> first, build());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> second, build());
    Data::Batch first_batch;
    Data::Batch second_batch;

    ASSERT_TRUE(SaTest::status_is_ok(first->read(0, first_batch)));
    ASSERT_TRUE(SaTest::status_is_ok(second->read(0, second_batch)));

    EXPECT_EQ(SaTest::read_tensor_bytes(first_batch.inputs),
              SaTest::read_tensor_bytes(second_batch.inputs));
}

/** @brief Tests that the lookup of an unknown type is refused with every type the registry has. */
TEST(Small_DatasetRegistry, AnUnknownDatasetTypeListsEveryTypeThisBuildHas)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Data::register_datasets(registry));

    SA_EXPECT_FAILURE(Data::find_dataset_row(registry, "coco"), StatusCode::InvalidDescription,
                      "coco", "dataset type", "'rings'");
}

/** @brief Tests that the lookup of a registered type gives its row. */
TEST(Small_DatasetRegistry, AKnownDatasetTypeIsFound)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Data::register_datasets(registry));

    SA_ASSERT_OK_AND_ASSIGN(const Data::DatasetEntry* const row,
                            Data::find_dataset_row(registry, "rings"));

    EXPECT_EQ(row, registry.find("rings"));
}

/** @brief Tests that an unknown key is refused by the reader of the options, before any dataset. */
TEST(Small_RingsOptions, AnUnknownDatasetKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_FAILURE(Data::read_rings_options(attributes_from(R"({"batch_sizes": 16})"), "data"),
                      StatusCode::InvalidDescription, "data", "'batch_sizes'", "'batch_size'");
}

TEST_F(Medium_Registries, AnUnknownOptimizerTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_FAILURE(Optim::make_optimizer(backend(), "lion", {}, attributes_from("{}"),
                                            "optimizer"),
                      StatusCode::InvalidDescription, "lion", "optimizer type", "'adamw'", "'sgd'");
}

/**
 * @brief Tests that the AdamW factory refuses a key it does not accept and names it.
 *
 * "momentum" is the key of SGD and "kernel" selects the reference update kernel, which a
 * configuration must not reach: a file that could set it would change what a run is.
 */
TEST_P(Medium_OptimizerKeys, AnUnknownOptimizerKeyIsRejectedByItsOwnFactory)
{
    const UnknownKeyCase& row = GetParam();

    SA_EXPECT_FAILURE(Optim::make_optimizer(backend(), "adamw", {}, attributes_from(row.attributes),
                                            "optimizer"),
                      StatusCode::InvalidDescription, "optimizer", row.quoted_key, row.detail);
}

INSTANTIATE_TEST_SUITE_P(
    Keys, Medium_OptimizerKeys,
    ::testing::Values(UnknownKeyCase{"Momentum", R"({"momentum": 0.9})", "'momentum'", "'beta1'"},
                      UnknownKeyCase{"Kernel", R"({"kernel": "separate"})", "'kernel'", "'kernel'"}),
    [](const ::testing::TestParamInfo<UnknownKeyCase>& info)
    {
        return std::string(info.param.name);
    });
