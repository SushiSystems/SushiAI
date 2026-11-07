/**************************************************************************/
/* test_replica_tensors.cpp                                               */
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
 * @file test_replica_tensors.cpp
 * @brief Tests ReplicaTensors::list(), the order of the tensors of a replica's packs.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <memory>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

#include "cluster_test_files.hpp"
#include "registries.hpp"
#include "replica_tensors.hpp"
#include "training_setup.hpp"

namespace
{
    using SushiAI::Test::require_value;
    using namespace SushiAI;
    using namespace SushiAI::Apps::ClusterTest;

    /** @brief Gives each case an engine on the shared test runtime. */
    class Medium_ReplicaTensors : public Test::EngineTest
    {

    };

    /**
     * @struct Listed
     * @brief Holds a training setup over the cluster test files and the tensors listed from it.
     */
    struct Listed
    {
        /** @brief Writes the files named @p name and builds the setup on @p backend. */
        Listed(const char* name, ExecutionBackend& backend)
            : files(name), registries(Apps::compose_registries().value()),
              setup(Apps::TrainingSetup::make(registries, files.spec, files.model_spec, backend,
                                              K_SEED)
                        .value()),
              tensors(Apps::ReplicaTensors::list(
                          SushiRuntime::span<const Optim::ParameterSlot>(
                              setup->parameters().data(), setup->parameters().size()),
                          setup->built(), setup->pool(),
                          SushiRuntime::span<const Train::BufferUpdateSlot>(
                              setup->updates().data(), setup->updates().size()))
                          .value())
        {

        }

        /** @brief The training file and the architecture file. */
        ClusterTestFiles files;

        /** @brief The registries the setup was made with. */
        Apps::Registries registries;

        /** @brief The setup whose parameters, loss and updates are listed. */
        std::unique_ptr<Apps::TrainingSetup> setup;

        /** @brief The listed tensors. */
        Apps::ReplicaTensors tensors;
    };

    /** @brief Expects the parameters and the first results to be the slots, in slot order. */
    void expect_parameters_and_gradients_in_slot_order(const Listed& listed)
    {
        const std::size_t parameters = listed.setup->parameters().size();
        ASSERT_EQ(listed.tensors.parameters.size(), parameters);
        ASSERT_GE(listed.tensors.results.size(), parameters);
        for (std::size_t index = 0; index < parameters; ++index)
        {
            EXPECT_EQ(listed.tensors.parameters[index].data(),
                      listed.setup->parameters()[index].parameter.data());
            EXPECT_EQ(listed.tensors.results[index].data(),
                      listed.setup->parameters()[index].gradient.data());
        }
    }

    /** @brief Expects each statistics source to follow the loss, in update order. */
    void expect_statistics_after_the_loss(const Listed& listed)
    {
        const std::size_t first = listed.setup->parameters().size() + 1;
        ASSERT_EQ(listed.tensors.results.size(), first + listed.setup->updates().size());
        for (std::size_t index = 0; index < listed.setup->updates().size(); ++index)
            EXPECT_EQ(listed.tensors.results[first + index].data(),
                      listed.setup->updates()[index].source.data());
    }
} // namespace

/** @brief Tests that the parameters and the first results are the slots, in slot order. */
TEST_F(Medium_ReplicaTensors, ListsTheParametersAndTheGradientsInSlotOrder)
{
    const Listed listed("sushiai_replica_tensors_slots.json", backend());

    expect_parameters_and_gradients_in_slot_order(listed);
}

/** @brief Tests that the loss follows the gradients in the results. */
TEST_F(Medium_ReplicaTensors, ListsTheLossAfterTheGradients)
{
    const Listed listed("sushiai_replica_tensors_loss.json", backend());

    const std::size_t parameters = listed.setup->parameters().size();
    ASSERT_GT(listed.tensors.results.size(), parameters);
    EXPECT_EQ(listed.tensors.results[parameters].data(),
              listed.setup->pool().tensor(listed.setup->built().spec.loss).data());
}

/** @brief Tests that the statistics sources follow the loss in the results, in update order. */
TEST_F(Medium_ReplicaTensors, ListsTheStatisticsAfterTheLoss)
{
    const Listed listed("sushiai_replica_tensors_statistics.json", backend());

    expect_statistics_after_the_loss(listed);
}
