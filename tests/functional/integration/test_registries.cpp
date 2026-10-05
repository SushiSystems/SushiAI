/****************************************************************/
/* test_registries.cpp                                          */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The two registries that need an Engine to exercise: datasets allocate device
// tensors, and optimizers allocate their state buffers. The layer and
// objective registries are host-lane and live in the unit tier.
//
// Rings gets a real test of its defining property - that it is NOT linearly
// separable - rather than only a shape check. A dataset whose whole reason to
// exist is "a single Linear layer cannot solve this" should fail if it ever
// quietly becomes solvable.

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"
#include "../../common/test_throws.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_Registries : public SaTest::EngineTest {};

    /** @brief Parses @p text as a bare attribute object. */
    Config::Value attributes_from(std::string_view text)
    {
        return Config::parse_json(text, "<test>");
    }
}

TEST_F(Integration_Registries, RingsIsBalancedAndCarriesItsDistractors)
{
    Data::RingsOptions options;
    options.batch_size = 32;
    options.batches = 4;
    const Data::Rings dataset(*engine, options);

    EXPECT_EQ(dataset.name(), "rings");
    EXPECT_EQ(dataset.input_dim(), 6);          // x, y and four distractors
    EXPECT_EQ(dataset.num_classes(), 4);
    EXPECT_EQ(dataset.batch_count(), 4u);

    // Round-robin assignment, so every batch is exactly balanced and accuracy
    // needs no class weighting to read correctly.
    std::vector<int> counts(4, 0);
    for (int32_t label : dataset.batch(0).labels) ++counts[static_cast<std::size_t>(label)];
    EXPECT_EQ(counts, (std::vector<int>{8, 8, 8, 8}));
}

TEST_F(Integration_Registries, RingsPlacesEachClassInItsOwnRadiusBand)
{
    // This is the property the whole problem rests on: the class is a function
    // of x^2 + y^2, which is degree 2 in the inputs and therefore not a
    // half-space. If a change ever made the radius bands overlap or the labels
    // track something linear, this fails rather than the accuracy quietly
    // rising in some other test.
    Data::RingsOptions options;
    options.batch_size = 64;
    options.batches = 1;
    options.position_noise = 0.0;               // no jitter, so the bands are exact
    const Data::Rings dataset(*engine, options);

    const Data::Batch& batch = dataset.batch(0);
    const float* x = batch.inputs.data_as<float>();
    const int64_t features = dataset.input_dim();

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
    const std::unique_ptr<Data::Dataset> dataset = Data::make_dataset(
        *engine, "rings",
        attributes_from(R"({"batch_size": 16, "batches": 2, "classes": 3,
                            "distractors": 1, "seed": 7})"),
        "data");

    EXPECT_EQ(dataset->batch_size(), 16);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->num_classes(), 3);
    EXPECT_EQ(dataset->input_dim(), 3);         // x, y and one distractor
}

TEST_F(Integration_Registries, TheSameSeedGivesTheSameData)
{
    // A host RNG rather than the engine's Philox stream, so reproducibility is
    // a property of the seed alone - the device stream mixes in a monotonic
    // offset counter, which is why dataset.hpp warns about set_seed.
    const auto build = [this]
    {
        return Data::make_dataset(*engine, "rings",
                                  attributes_from(R"({"batch_size": 8, "batches": 1, "seed": 42})"),
                                  "data");
    };

    const std::unique_ptr<Data::Dataset> first = build();
    const std::unique_ptr<Data::Dataset> second = build();

    const float* a = first->batch(0).inputs.data_as<float>();
    const float* b = second->batch(0).inputs.data_as<float>();
    for (int64_t i = 0; i < first->batch(0).inputs.num_elements; ++i)
        EXPECT_EQ(a[i], b[i]) << "element " << i;
}

TEST_F(Integration_Registries, AnUnknownDatasetTypeListsEveryTypeThisBuildHas)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(*engine, "coco", attributes_from("{}"), "data"),
        "coco", "dataset type", "'rings'");
}

TEST_F(Integration_Registries, AnUnknownDatasetKeyIsRejectedByItsOwnFactory)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(*engine, "rings",
                                 attributes_from(R"({"batch_sizes": 16})"), "data"),
        "data", "'batch_sizes'", "'batch_size'");
}

TEST_F(Integration_Registries, ANonPositiveRingsExtentIsRefused)
{
    SA_EXPECT_ERROR(
        (void)Data::make_dataset(*engine, "rings",
                                 attributes_from(R"({"batch_size": 0})"), "data"),
        "positive");
}

TEST_F(Integration_Registries, TheOptimizerRegistryBuildsBothRulesFromAConfigValue)
{
    // One real parameter/gradient pair, so the optimizers allocate their state
    // against something rather than against an empty list.
    const int64_t dims[1] = {4};
    Optim::ParameterSlot slot;
    slot.name = "0.weight";
    slot.parameter = allocate_tensor(*engine, SushiRuntime::span<const int64_t>(dims, 1));
    slot.gradient = allocate_tensor(*engine, SushiRuntime::span<const int64_t>(dims, 1));

    const std::unique_ptr<Optim::Optimizer> adam = Optim::make_optimizer(
        *engine, "adamw", {slot},
        attributes_from(R"({"learning_rate": 0.003, "weight_decay": 0.02})"), "optimizer");
    EXPECT_TRUE(adam->is_replay_invariant());

    const std::unique_ptr<Optim::Optimizer> sgd = Optim::make_optimizer(
        *engine, "sgd", {slot},
        attributes_from(R"({"learning_rate": 0.1, "momentum": 0.9, "nesterov": true})"),
        "optimizer");
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
    // "momentum" is SGD's, not AdamW's, and asking AdamW for it must not be a
    // line that quietly does nothing.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"momentum": 0.9})"), "optimizer"),
        "optimizer", "'momentum'", "'beta1'");
}

TEST_F(Integration_Registries, AConfigurationCannotReachTheReferenceUpdateKernel)
{
    // UpdateKernel selects the spelling a byte-for-byte regression test
    // compares the fused kernel against; SEPARATE additionally cannot join the
    // compiled plan. A config file that could set it would be able to change
    // what a run IS, so "kernel" is not an accepted key.
    SA_EXPECT_ERROR(
        (void)Optim::make_optimizer(*engine, "adamw", {},
                                    attributes_from(R"({"kernel": "separate"})"), "optimizer"),
        "'kernel'");
}
