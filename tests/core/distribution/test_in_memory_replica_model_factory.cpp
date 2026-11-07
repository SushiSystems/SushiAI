/**************************************************************************/
/* test_in_memory_replica_model_factory.cpp                               */
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
 * @file test_in_memory_replica_model_factory.cpp
 * @brief Tests that InMemoryReplicaModelFactory keeps its calls and returns what its inner
 *        factory returns.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include <SushiAI/distribution/in_memory_replica_model.hpp>
#include <SushiAI/distribution/in_memory_replica_model_factory.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;

    /** @brief Holds the seed of the draws of a case. */
    constexpr std::uint64_t K_SEED = 11;

    /** @brief Returns a factory that builds a model of one parameter and one result. */
    ReplicaModelFactory one_by_one()
    {
        return [](ExecutionBackend&, const ReplicaDraws&, std::int64_t)
            -> Result<std::unique_ptr<ReplicaModel>>
        {
            return Result<std::unique_ptr<ReplicaModel>>(
                std::make_unique<InMemoryReplicaModel>(1, 1));
        };
    }

    /** @brief Returns a factory that builds nothing and fails. */
    ReplicaModelFactory refusing()
    {
        return [](ExecutionBackend&, const ReplicaDraws&, std::int64_t)
            -> Result<std::unique_ptr<ReplicaModel>>
        {
            return Result<std::unique_ptr<ReplicaModel>>(Status::from_message(
                StatusCode::InvalidDescription, [] { return std::string("no model here"); }));
        };
    }
} // namespace

/** @brief Tests that a fresh factory has received no call. */
TEST(Small_InMemoryReplicaModelFactory, StartsEmpty)
{
    const InMemoryReplicaModelFactory factory(one_by_one());

    EXPECT_TRUE(factory.calls().empty());
}

/** @brief Tests that the draws and rows of each call are kept in order through the factory. */
TEST(Small_InMemoryReplicaModelFactory, KeepsTheDrawsAndRowsOfEachCallInOrder)
{
    HostExecutionBackend backend;
    InMemoryReplicaModelFactory memory(one_by_one());
    const ReplicaModelFactory port = memory.as_factory();

    EXPECT_TRUE(port(backend, ReplicaDraws{K_SEED, 0, 1}, 3).has_value());
    EXPECT_TRUE(port(backend, ReplicaDraws{K_SEED, 1, 1}, 5).has_value());

    ASSERT_EQ(memory.calls().size(), 2u);
    EXPECT_EQ(memory.calls()[0].draws, (ReplicaDraws{K_SEED, 0, 1}));
    EXPECT_EQ(memory.calls()[0].rows, 3);
    EXPECT_EQ(memory.calls()[1].draws, (ReplicaDraws{K_SEED, 1, 1}));
    EXPECT_EQ(memory.calls()[1].rows, 5);
}

/** @brief Tests that the model the inner factory builds is the one the call returns. */
TEST(Small_InMemoryReplicaModelFactory, ReturnsTheModelOfTheInnerFactory)
{
    HostExecutionBackend backend;
    InMemoryReplicaModelFactory memory(one_by_one());

    Result<std::unique_ptr<ReplicaModel>> built = memory.make(backend, ReplicaDraws{K_SEED, 0, 1}, 2);

    ASSERT_TRUE(built.has_value());
    EXPECT_EQ(built.value()->parameters().size(), 1u);
}

/** @brief Tests that a failure of the inner factory is returned and the call is kept. */
TEST(Small_InMemoryReplicaModelFactory, ReturnsTheFailureOfTheInnerFactoryAndKeepsTheCall)
{
    HostExecutionBackend backend;
    InMemoryReplicaModelFactory memory(refusing());

    SA_EXPECT_FAILURE(memory.make(backend, ReplicaDraws{K_SEED, 0, 1}, 2),
                      StatusCode::InvalidDescription, "no model here");
    EXPECT_EQ(memory.calls().size(), 1u);
}
