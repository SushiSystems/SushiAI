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

    /** @brief Expects every parameter slot of @p setup to hold a gradient with data. */
    void expect_every_gradient_has_data(const Apps::TrainingSetup& setup)
    {
        for (const Optim::ParameterSlot& slot : setup.parameters())
            EXPECT_NE(slot.gradient.data(), nullptr) << slot.name;
    }
} // namespace

/** @brief Tests that a setup without rows traces the data's batch. */
TEST_F(Medium_TrainingSetup, TracesTheDataBatchWhenNoRowsAreGiven)
{
    const ClusterTestFiles files("sushiai_training_setup_batch.json");
    const Apps::Registries registries = Apps::compose_registries().value();

    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, backend(), K_SEED)
            .value();

    EXPECT_EQ(setup->given().batch, 4);
    EXPECT_EQ(setup->dataset().batch_size(), 4);
}

/** @brief Tests that a setup binds the parameters and the state of the model. */
TEST_F(Medium_TrainingSetup, BindsTheParametersAndTheStateOfTheModel)
{
    const ClusterTestFiles files("sushiai_training_setup_bound.json");
    const Apps::Registries registries = Apps::compose_registries().value();

    const std::unique_ptr<Apps::TrainingSetup> setup =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, backend(), K_SEED)
            .value();

    EXPECT_FALSE(setup->parameters().empty());
    EXPECT_FALSE(setup->state().entries().empty());
}

/** @brief Tests that the rows a replica is given are the batch the graph is traced over. */
TEST(Small_TrainingSetup, TracesTheRowsItIsGiven)
{
    EXPECT_EQ(Apps::trace_batch(2, 4), 2);
}

/** @brief Tests that, with no rows given, the graph is traced over the data's batch. */
TEST(Small_TrainingSetup, TracesTheDataBatchWhenNoRowsAreGiven)
{
    EXPECT_EQ(Apps::trace_batch(std::nullopt, 4), 4);
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
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, backend(), K_SEED)
            .value();
    const std::unique_ptr<Apps::TrainingSetup> results =
        Apps::TrainingSetup::make(registries, files.spec, files.model_spec, backend(), K_SEED,
                                  std::nullopt, Apps::SetupScope::Results)
            .value();

    const Train::TrainingGraph& built = results->built();
    EXPECT_FALSE(results->pool().has_plan());
    EXPECT_TRUE(results->pool().is_bound(built.spec.loss));
    EXPECT_FALSE(results->pool().is_bound(built.spec.inputs));
    ASSERT_EQ(results->parameters().size(), whole->parameters().size());
    expect_every_gradient_has_data(*results);

    EXPECT_TRUE(whole->pool().has_plan());
    EXPECT_LT(results->pool().allocated_bytes(), whole->pool().allocated_bytes());
}

/** @brief Tests that an architecture of another input shape than the data's is refused. */
TEST(Small_TrainingSetup, RefusesAModelOfAnotherInputShape)
{
    SA_EXPECT_FAILURE(Apps::check_input_shape("model.json", Shape{5}, "rings", Shape{2}),
                      StatusCode::InvalidDescription, "model.json", "rings");
}

/** @brief Tests that an architecture of the data's input shape is accepted. */
TEST(Small_TrainingSetup, AcceptsAModelOfTheDataInputShape)
{
    SA_EXPECT_OK(Apps::check_input_shape("model.json", Shape{2}, "rings", Shape{2}));
}
