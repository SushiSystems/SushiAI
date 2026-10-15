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
 * @file test_dataset_registry.cpp
 * @brief Tests the dataset registry: its entries, the roles and targets they declare, and what it
 *        refuses.
 * @author Mustafa Garip
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>
#include <SushiAI/model_text/model_text.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief The "rings" row, as register_datasets adds it. */
    constexpr Data::DatasetEntry K_RINGS_ROW{"rings", &Data::make_rings};

    /** @brief A second row, under a type no module registers. */
    constexpr Data::DatasetEntry K_SECOND_ROW{"moons", &Data::make_rings};
} // namespace

/** @brief Tests that rows() returns the rows in the order add() received them. */
TEST(Medium_DatasetRegistry, AddKeepsInsertionOrder)
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
TEST(Medium_DatasetRegistry, FindReturnsNullForAnUnknownType)
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
TEST(Medium_DatasetRegistry, ATypeAddedTwiceIsRefused)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(registry.add(K_RINGS_ROW));

    SA_EXPECT_FAILURE(registry.add(K_RINGS_ROW), StatusCode::InvalidDescription, "'rings'",
                      "already registered");
}

/** @brief Tests that register_datasets adds the "rings" row and no other. */
TEST(Medium_DatasetRegistry, RegisterDatasetsAddsRingsOnly)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Data::register_datasets(registry));

    const SushiRuntime::span<const Data::DatasetEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].type, "rings");
    EXPECT_EQ(registry.find("mnist"), nullptr);
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

    /** @brief Returns the class count the labels role of @p dataset states, or -1. */
    int64_t class_count(const Data::Dataset& dataset)
    {
        const Data::TargetDescription* labels =
            Data::find_target(dataset.targets(), Data::K_ROLE_LABELS);
        return labels != nullptr ? labels->classes : -1;
    }

    /** @brief Returns the class index of every sample of @p batch, or nothing. */
    std::vector<int32_t> labels_of(const Data::Batch& batch)
    {
        const Data::BatchTarget* labels = Data::find_target(batch.targets, Data::K_ROLE_LABELS);
        if (labels == nullptr || labels->values.dtype != DType::Int32)
            return {};

        const int32_t* values = labels->values.data_as<int32_t>();
        return std::vector<int32_t>(values, values + labels->values.num_elements);
    }
}

TEST_F(Medium_Registries, RingsIsBalancedAndCarriesItsDistractors)
{
    Data::RingsOptions options;
    options.batch_size = 32;
    options.batches = 4;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    EXPECT_EQ(dataset->name(), "rings");
    EXPECT_EQ(dataset->sample_shape(), Shape{6});   // x, y and four distractors
    EXPECT_EQ(class_count(*dataset), 4);
    EXPECT_EQ(dataset->batch_count(), 4u);

    // Round-robin labels balance every batch; see tests/core/data/README.md.
    std::vector<int> counts(4, 0);
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    const std::vector<int32_t> labels = labels_of(dataset_batch);
    ASSERT_EQ(labels.size(), 32u);
    for (int32_t label : labels) ++counts[static_cast<std::size_t>(label)];
    EXPECT_EQ(counts, (std::vector<int>{8, 8, 8, 8}));
}

/** @brief Tests that Rings publishes one role: labels, one int32 per sample over its classes. */
TEST_F(Medium_Registries, RingsPublishesItsLabelsAsOneIndexPerSample)
{
    Data::RingsOptions options;
    options.batch_size = 8;
    options.batches = 1;
    options.classes = 3;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    ASSERT_EQ(dataset->targets().size(), 1u);
    const Data::TargetDescription& described = dataset->targets().front();
    EXPECT_EQ(described.role, Data::K_ROLE_LABELS);
    EXPECT_EQ(described.sample_shape, Shape{});
    EXPECT_EQ(described.dtype, DType::Int32);
    EXPECT_EQ(described.classes, 3);

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    ASSERT_EQ(batch.targets.size(), 1u);
    EXPECT_EQ(batch.targets.front().role, Data::K_ROLE_LABELS);
    EXPECT_EQ(batch.targets.front().values.dtype, DType::Int32);
    ASSERT_EQ(batch.targets.front().values.rank, 1);
    EXPECT_EQ(batch.targets.front().values.shape[0], 8);
    EXPECT_EQ(labels_of(batch), (std::vector<int32_t>{0, 1, 2, 0, 1, 2, 0, 1}));
}

TEST_F(Medium_Registries, RingsReportItsSampleShape)
{
    Data::RingsOptions options;
    options.batch_size = 8;
    options.batches = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    // The default options add four distractors to the two ring coordinates.
    EXPECT_EQ(dataset->sample_shape(), Shape{6});

    // The batch axis is prepended: [rows, sample...], here [8, 6].
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    const SushiBLAS::Tensor& inputs = dataset_batch.inputs;
    EXPECT_EQ(inputs.rank, 2);
    EXPECT_EQ(inputs.shape[0], 8);
    EXPECT_EQ(inputs.shape[1], 6);
}

TEST_F(Medium_Registries, RingsPlacesEachClassInItsOwnRadiusBand)
{
    // The class is a function of x^2 + y^2; see tests/core/data/README.md.
    Data::RingsOptions options;
    options.batch_size = 64;
    options.batches = 1;
    options.position_noise = 0.0;               // no jitter, so the bands are exact
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, batch)));
    const float* x = batch.inputs.data_as<float>();
    const int64_t features = dataset->sample_shape().num_elements();
    const std::vector<int32_t> labels = labels_of(batch);
    ASSERT_EQ(labels.size(), 64u);

    for (std::size_t r = 0; r < labels.size(); ++r)
    {
        const double px = x[static_cast<int64_t>(r) * features];
        const double py = x[static_cast<int64_t>(r) * features + 1];
        const double radius = std::sqrt(px * px + py * py);
        const double band = static_cast<double>(labels[r]);

        EXPECT_GE(radius, band - 1e-4) << "row " << r;
        EXPECT_LT(radius, band + options.ring_width + 1e-4) << "row " << r;
    }
}

TEST_F(Medium_Registries, TheDatasetRegistryBuildsRingsFromAConfigValue)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        Data::make_dataset(datasets, *engine, "rings",
                           attributes_from(R"({"batch_size": 16, "batches": 2, "classes": 3,
                            "distractors": 1, "seed": 7})"),
                           "data"));

    EXPECT_EQ(dataset->batch_size(), 16);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(class_count(*dataset), 3);
    EXPECT_EQ(dataset->sample_shape(), Shape{3});   // x, y and one distractor
}

TEST_F(Medium_Registries, TheSameSeedGivesTheSameData)
{
    // Data comes from a host RNG, not Philox; see tests/core/data/README.md.
    const auto build = [this]
    {
        return Data::make_dataset(datasets, *engine, "rings",
                                  attributes_from(R"({"batch_size": 8, "batches": 1, "seed": 42})"),
                                  "data");
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> first, build());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> second, build());

    Data::Batch first_batch;
    ASSERT_TRUE(SaTest::status_is_ok(first->read(0, first_batch)));
    const float* a = first_batch.inputs.data_as<float>();
    Data::Batch second_batch;
    ASSERT_TRUE(SaTest::status_is_ok(second->read(0, second_batch)));
    const float* b = second_batch.inputs.data_as<float>();
    for (int64_t i = 0; i < first_batch.inputs.num_elements; ++i)
        EXPECT_EQ(a[i], b[i]) << "element " << i;
}

TEST_F(Medium_Registries, AnUnknownDatasetTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "coco", attributes_from("{}"), "data"),
        StatusCode::InvalidDescription, "coco", "dataset type", "'rings'");
}

TEST_F(Medium_Registries, AnUnknownDatasetKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(datasets, *engine, "rings",
                                 attributes_from(R"({"batch_sizes": 16})"), "data"),
        "data", "'batch_sizes'", "'batch_size'");
}

TEST_F(Medium_Registries, ANonPositiveRingsExtentIsRefused)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "rings",
                           attributes_from(R"({"batch_size": 0})"), "data"),
        StatusCode::InvalidDescription, "positive");
}

TEST_F(Medium_Registries, TheOptimizerRegistryBuildsBothRulesFromAConfigValue)
{
    // One real slot backs the optimizer state; see tests/core/data/README.md.
    const int64_t dims[1] = {4};
    const SushiRuntime::span<const int64_t> shape(dims, 1);
    Optim::ParameterSlot slot;
    slot.name = "0.weight";
    SA_ASSERT_OK_AND_ASSIGN(slot.parameter, allocate_tensor(*engine, shape));
    SA_ASSERT_OK_AND_ASSIGN(slot.gradient, allocate_tensor(*engine, shape));

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> adam,
        Optim::make_optimizer(
            *engine, "adamw", {slot},
            attributes_from(R"({"learning_rate": 0.003, "weight_decay": 0.02})"),
            "optimizer"));
    EXPECT_TRUE(adam->is_replay_invariant());

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Optim::Optimizer> sgd,
        Optim::make_optimizer(
            *engine, "sgd", {slot},
            attributes_from(R"({"learning_rate": 0.1, "momentum": 0.9, "nesterov": true})"),
            "optimizer"));
    EXPECT_TRUE(sgd->is_replay_invariant());
}

TEST_F(Medium_Registries, AnUnknownOptimizerTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "lion", {}, attributes_from("{}"), "optimizer"),
        "lion", "optimizer type", "'adamw'", "'sgd'");
}

TEST_F(Medium_Registries, AnUnknownOptimizerKeyIsRejectedByItsOwnFactory)
{
    // "momentum" is SGD's key, not AdamW's; see tests/core/data/README.md.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"momentum": 0.9})"), "optimizer"),
        "optimizer", "'momentum'", "'beta1'");
}

TEST_F(Medium_Registries, AConfigurationCannotReachTheReferenceUpdateKernel)
{
    // "kernel" is not an accepted config key; see tests/core/data/README.md.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"kernel": "separate"})"), "optimizer"),
        "'kernel'");
}
