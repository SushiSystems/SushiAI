/**************************************************************************/
/* test_registries.cpp                                                    */
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
 * @file test_registries.cpp
 * @brief Tests the dataset and optimizer registries, which need an Engine.
 * @author Mustafa Garip
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset_registry.hpp>
#include <common/model_text.hpp>
#include <common/test_utils.hpp>
#include <common/test_status.hpp>
#include <common/test_throws.hpp>

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Registries : public SaTest::EngineTest
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
}

TEST_F(Integration_Registries, RingsIsBalancedAndCarriesItsDistractors)
{
    Data::RingsOptions options;
    options.batch_size = 32;
    options.batches = 4;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    EXPECT_EQ(dataset->name(), "rings");
    EXPECT_EQ(dataset->input_dim(), 6);         // x, y and four distractors
    EXPECT_EQ(dataset->num_classes(), 4);
    EXPECT_EQ(dataset->batch_count(), 4u);

    // Round-robin labels balance every batch; see tests/functional/integration/README.md.
    std::vector<int> counts(4, 0);
    for (int32_t label : dataset->batch(0).labels) ++counts[static_cast<std::size_t>(label)];
    EXPECT_EQ(counts, (std::vector<int>{8, 8, 8, 8}));
}

TEST_F(Integration_Registries, RingsReportItsSampleShape)
{
    Data::RingsOptions options;
    options.batch_size = 8;
    options.batches = 1;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    // The default options add four distractors to the two ring coordinates.
    EXPECT_EQ(dataset->sample_shape(), Shape{6});
    EXPECT_EQ(dataset->sample_shape().num_elements(), dataset->input_dim());

    // The batch axis is prepended: [rows, sample...], here [8, 6].
    const SushiBLAS::Tensor& inputs = dataset->batch(0).inputs;
    EXPECT_EQ(inputs.rank, 2);
    EXPECT_EQ(inputs.shape[0], 8);
    EXPECT_EQ(inputs.shape[1], 6);
}

TEST_F(Integration_Registries, RingsPlacesEachClassInItsOwnRadiusBand)
{
    // The class is a function of x^2 + y^2; see tests/functional/integration/README.md.
    Data::RingsOptions options;
    options.batch_size = 64;
    options.batches = 1;
    options.position_noise = 0.0;               // no jitter, so the bands are exact
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Rings> dataset,
                            Data::Rings::make(*engine, options));

    const Data::Batch& batch = dataset->batch(0);
    const float* x = batch.inputs.data_as<float>();
    const int64_t features = dataset->input_dim();

    for (std::size_t r = 0; r < batch.labels.size(); ++r)
    {
        const double px = x[static_cast<int64_t>(r) * features];
        const double py = x[static_cast<int64_t>(r) * features + 1];
        const double radius = std::sqrt(px * px + py * py);
        const double band = static_cast<double>(batch.labels[r]);

        EXPECT_GE(radius, band - 1e-4) << "row " << r;
        EXPECT_LT(radius, band + options.ring_width + 1e-4) << "row " << r;
    }
}

TEST_F(Integration_Registries, TheDatasetRegistryBuildsRingsFromAConfigValue)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Data::Dataset> dataset,
        Data::make_dataset(datasets, *engine, "rings",
                           attributes_from(R"({"batch_size": 16, "batches": 2, "classes": 3,
                            "distractors": 1, "seed": 7})"),
                           "data"));

    EXPECT_EQ(dataset->batch_size(), 16);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->num_classes(), 3);
    EXPECT_EQ(dataset->input_dim(), 3);         // x, y and one distractor
}

TEST_F(Integration_Registries, TheSameSeedGivesTheSameData)
{
    // Data comes from a host RNG, not Philox; see tests/functional/integration/README.md.
    const auto build = [this]
    {
        return Data::make_dataset(datasets, *engine, "rings",
                                  attributes_from(R"({"batch_size": 8, "batches": 1, "seed": 42})"),
                                  "data");
    };

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> first, build());
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::Dataset> second, build());

    const float* a = first->batch(0).inputs.data_as<float>();
    const float* b = second->batch(0).inputs.data_as<float>();
    for (int64_t i = 0; i < first->batch(0).inputs.num_elements; ++i)
        EXPECT_EQ(a[i], b[i]) << "element " << i;
}

TEST_F(Integration_Registries, AnUnknownDatasetTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "coco", attributes_from("{}"), "data"),
        StatusCode::InvalidDescription, "coco", "dataset type", "'rings'");
}

TEST_F(Integration_Registries, AnUnknownDatasetKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(datasets, *engine, "rings",
                                 attributes_from(R"({"batch_sizes": 16})"), "data"),
        "data", "'batch_sizes'", "'batch_size'");
}

TEST_F(Integration_Registries, ANonPositiveRingsExtentIsRefused)
{
    SA_EXPECT_FAILURE(
        Data::make_dataset(datasets, *engine, "rings",
                           attributes_from(R"({"batch_size": 0})"), "data"),
        StatusCode::InvalidDescription, "positive");
}

TEST_F(Integration_Registries, TheOptimizerRegistryBuildsBothRulesFromAConfigValue)
{
    // One real slot backs the optimizer state; see tests/functional/integration/README.md.
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

TEST_F(Integration_Registries, AnUnknownOptimizerTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "lion", {}, attributes_from("{}"), "optimizer"),
        "lion", "optimizer type", "'adamw'", "'sgd'");
}

TEST_F(Integration_Registries, AnUnknownOptimizerKeyIsRejectedByItsOwnFactory)
{
    // "momentum" is SGD's key, not AdamW's; see tests/functional/integration/README.md.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"momentum": 0.9})"), "optimizer"),
        "optimizer", "'momentum'", "'beta1'");
}

TEST_F(Integration_Registries, AConfigurationCannotReachTheReferenceUpdateKernel)
{
    // "kernel" is not an accepted config key; see tests/functional/integration/README.md.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"kernel": "separate"})"), "optimizer"),
        "'kernel'");
}
