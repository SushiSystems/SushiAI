/**************************************************************************/
/* test_replica_model.cpp                                                 */
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
 * @file test_replica_model.cpp
 * @brief Tests what a ReplicaModelFactory is called with, and what its model must hold.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include <SushiAI/distribution/in_memory_replica_model.hpp>
#include <SushiAI/distribution/in_memory_replica_model_factory.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;
} // namespace

/** @brief Tests that the factory runs once, with the step's draws and micro-batch rows. */
TEST(Medium_ReplicaModel, TheFactoryReceivesTheDrawsAndTheMicroBatchRows)
{
    InMemoryReplicaModelFactory factory(linear_factory());
    const ReplicaDraws draws{K_SEED, 0, K_RUN};
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(factory.as_factory(), draws, 3));

    ASSERT_EQ(factory.calls().size(), 1u);
    EXPECT_EQ(factory.calls().front().draws, draws);
    EXPECT_EQ(factory.calls().front().rows, 3);
    EXPECT_EQ(step->draws(), draws);
}

/** @brief Tests that a step refuses a model that packs no result. */
TEST(Medium_ReplicaModel, AStepRefusesAModelWithNoResult)
{
    const ReplicaModelFactory factory = [](ExecutionBackend&, const ReplicaDraws&, std::int64_t)
        -> Result<std::unique_ptr<ReplicaModel>>
    {
        return Result<std::unique_ptr<ReplicaModel>>(std::make_unique<InMemoryReplicaModel>(1, 0));
    };

    SA_EXPECT_FAILURE(ReplicaStep::make(factory, ReplicaDraws{K_SEED, 0, K_RUN}, 2),
                      StatusCode::InvalidDescription, "no result to pack");
}
