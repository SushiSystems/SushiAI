/**************************************************************************/
/* test_training_setup.cpp                                                */
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
 * @file test_training_setup.cpp
 * @brief Tests TrainingSetup, which builds what a training file describes on one engine.
 * @author Mustafa Garip
 */

#include <memory>
#include <optional>
#include <gtest/gtest.h>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/training_graph.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "training_setup.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_TrainingSetup : public Test::EngineTest
    {

    };
} // namespace

/** @brief Tests that a setup without rows traces the data's batch and binds every parameter. */
TEST_F(Medium_TrainingSetup, TracesTheDataBatchWhenNoRowsAreGiven)
{
    const ClusterTestFiles files("sushiai_training_setup_batch.json");
    const Apps::Registries registries = Apps::compose_registries().value();

    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, *engine, K_SEED)
            .value();

    EXPECT_EQ(setup->given().batch, 4);
    EXPECT_FALSE(setup->parameters().empty());
    EXPECT_FALSE(setup->state().entries().empty());
    EXPECT_EQ(setup->dataset().batch_size(), 4);
}

/** @brief Tests that a setup traces the rows it is given, while the data keeps its batch. */
TEST_F(Medium_TrainingSetup, TracesTheRowsItIsGiven)
{
    const ClusterTestFiles files("sushiai_training_setup_rows.json");
    const Apps::Registries registries = Apps::compose_registries().value();

    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, *engine, K_SEED, 2)
            .value();

    EXPECT_EQ(setup->given().batch, 2);
    EXPECT_EQ(setup->dataset().batch_size(), 4);
}

/**
 * @brief Tests that a setup of the results scope binds the parameters, the gradients and the
 *        loss, plans no memory and allocates fewer bytes than a setup of the step scope.
 */
TEST_F(Medium_TrainingSetup, HoldsOnlyTheResultsOfTheStepInTheResultsScope)
{
    const ClusterTestFiles files("sushiai_training_setup_results.json");
    const Apps::Registries registries = Apps::compose_registries().value();

    const std::unique_ptr<Apps::TrainingSetup> whole =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, *engine, K_SEED)
            .value();
    const std::unique_ptr<Apps::TrainingSetup> results =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, *engine, K_SEED,
                                  std::nullopt, Apps::SetupScope::Results)
            .value();

    const Train::TrainingGraph& built = results->built();
    EXPECT_FALSE(results->pool().has_plan());
    EXPECT_TRUE(results->pool().is_bound(built.spec.loss));
    EXPECT_FALSE(results->pool().is_bound(built.spec.inputs));
    ASSERT_EQ(results->parameters().size(), whole->parameters().size());
    for (const Optim::ParameterSlot& slot : results->parameters())
        EXPECT_NE(slot.gradient.data(), nullptr) << slot.name;

    EXPECT_TRUE(whole->pool().has_plan());
    EXPECT_LT(results->pool().allocated_bytes(), whole->pool().allocated_bytes());
}

/** @brief Tests that an architecture of another input shape than the data's is refused. */
TEST_F(Medium_TrainingSetup, RefusesAModelOfAnotherInputShape)
{
    const ClusterTestFiles files("sushiai_training_setup_shape.json");
    const Apps::Registries registries = Apps::compose_registries().value();
    Config::ModelSpec other = files.model_spec;
    other.input_shape = Shape{5};

    SA_EXPECT_FAILURE(
        Apps::TrainingSetup::make(registries, files.spec, other, *engine, K_SEED).status(),
        StatusCode::InvalidDescription);
}
