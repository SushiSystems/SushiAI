/**************************************************************************/
/* test_replica_step.cpp                                                  */
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
 * @file test_replica_step.cpp
 * @brief Tests that ReplicaStep writes the results of its micro-batches, and their mean.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_model.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;

    /** @brief Holds the bytes of one pack, in memory the host owns. */
    struct HostPack
    {
        /** @brief Allocates a zeroed buffer of the layout's size. */
        explicit HostPack(const SushiBLAS::PackLayout& layout)
            : bytes(layout.size_bytes()), bound(bytes.data(), layout), layout(&layout)
        {

        }

        /** @brief Returns the bytes of entry @p index. */
        std::vector<unsigned char> entry(std::size_t index) const
        {
            const SushiBLAS::PackEntry& placed = layout->entries()[index];
            const unsigned char* const first =
                reinterpret_cast<const unsigned char*>(bytes.data()) + placed.offset_bytes;
            return std::vector<unsigned char>(first, first + placed.size_bytes);
        }

        std::vector<std::byte> bytes;        /**< @brief Holds the pack. */
        SushiBLAS::BoundPack bound;          /**< @brief Names the pack's tensors. */
        const SushiBLAS::PackLayout* layout; /**< @brief The pack's layout; outlives this. */
    };

    /** @brief Writes parameter_value() into every tensor of a parameter pack. */
    void fill_parameters(const SushiBLAS::PackLayout& layout, HostPack& pack)
    {
        std::size_t element = 0;
        for (std::size_t index = 0; index < layout.entries().size(); ++index)
        {
            float* const values = pack.bound.tensor(index).as<float>();
            for (std::size_t value = 0; value < layout.entries()[index].element_count; ++value)
                values[value] = parameter_value(element++);
        }
    }

    /** @brief Returns the rows [first, first + count) of the first batch of replica 0. */
    ReplicaRows rows_from(std::uint64_t first, std::uint64_t count, std::uint64_t micro_batch)
    {
        ReplicaRows rows;
        rows.micro_batch_index = micro_batch;
        rows.rows = RowPositions{first, count};
        rows.draws = ReplicaDraws{K_SEED, 0, K_RUN};
        return rows;
    }

    /** @brief Returns the float32 elements of a result's bytes. */
    std::vector<float> as_floats(const std::vector<unsigned char>& bytes)
    {
        std::vector<float> values(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        return values;
    }

    /** @brief Expects every byte of every result entry of @p results to be zero. */
    void expect_results_zero(const HostPack& results, const SushiBLAS::PackLayout& layout)
    {
        for (std::size_t index = 0; index < layout.entries().size(); ++index)
        {
            const std::vector<unsigned char> bytes = results.entry(index);
            EXPECT_EQ(bytes, std::vector<unsigned char>(bytes.size(), 0)) << "result " << index;
        }
    }

    /** @brief Expects each entry of @p results to equal the same entry of @p expected bit for bit. */
    void expect_results_equal(const HostPack& results,
                              const std::vector<std::vector<unsigned char>>& expected)
    {
        ASSERT_EQ(results.layout->entries().size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index)
            EXPECT_EQ(results.entry(index), expected[index]) << "result " << index;
    }

    /** @brief Expects each entry of @p results to hold (a + b) / 2 in float32, bit for bit. */
    void expect_results_mean_of(const HostPack& results,
                                const std::vector<std::vector<unsigned char>>& first,
                                const std::vector<std::vector<unsigned char>>& second)
    {
        ASSERT_EQ(first.size(), second.size());
        for (std::size_t index = 0; index < first.size(); ++index)
        {
            const std::vector<float> got = as_floats(results.entry(index));
            const std::vector<float> a = as_floats(first[index]);
            const std::vector<float> b = as_floats(second[index]);
            ASSERT_EQ(got.size(), a.size()) << "result " << index;
            for (std::size_t element = 0; element < got.size(); ++element)
            {
                const float expected = (a[element] + b[element]) / 2.0f;
                EXPECT_EQ(std::memcmp(&got[element], &expected, sizeof(float)), 0)
                    << "result " << index << ", element " << element;
            }
        }
    }
} // namespace

/** @brief Tests that make() refuses a micro-batch of no row before it creates a runtime. */
TEST(Small_ReplicaStep, MakeRefusesAMicroBatchBelowOne)
{
    SA_EXPECT_FAILURE(ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN}, 0),
                      StatusCode::InvalidDescription, "holds 0 rows");
}

namespace
{
    /** @brief Gives each case the one backend of the program. */
    class Medium_ReplicaStep : public SushiAI::Test::EngineTest {};
} // namespace

/** @brief Tests that a step that ran no micro-batch writes zeros. */
TEST_F(Medium_ReplicaStep, NoMicroBatchWritesZeros)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN},
                                              4));
    HostPack parameters(step->parameter_layout());
    HostPack results(step->result_layout());
    std::memset(results.bytes.data(), 0xFF, results.bytes.size());
    fill_parameters(step->parameter_layout(), parameters);

    SA_EXPECT_OK(step->begin(parameters.bound));
    SA_EXPECT_OK(step->finish(results.bound));

    expect_results_zero(results, step->result_layout());
}

/** @brief Tests that one micro-batch writes the replay's results byte for byte. */
TEST_F(Medium_ReplicaStep, OneMicroBatchWritesTheReplayResultsBitForBit)
{
    const SingleMachineStep reference(backend(), 4);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN},
                                              4));
    HostPack parameters(step->parameter_layout());
    HostPack results(step->result_layout());
    fill_parameters(step->parameter_layout(), parameters);

    SA_EXPECT_OK(step->begin(parameters.bound));
    SA_EXPECT_OK(step->run_micro_batch(rows_from(0, 4, 0)));
    SA_EXPECT_OK(step->finish(results.bound));

    expect_results_equal(results, reference.results);
}

/**
 * @brief Tests that two micro-batches write (a + b) / 2 of the two replays, in float32 and in
 *        micro-batch order, bit for bit.
 */
TEST_F(Medium_ReplicaStep, TwoMicroBatchesWriteTheirMeanInMicroBatchOrder)
{
    const SingleMachineStep first(backend(), 2, 0);
    const SingleMachineStep second(backend(), 2, 2);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN},
                                              2));
    HostPack parameters(step->parameter_layout());
    HostPack results(step->result_layout());
    fill_parameters(step->parameter_layout(), parameters);

    SA_EXPECT_OK(step->begin(parameters.bound));
    SA_EXPECT_OK(step->run_micro_batch(rows_from(0, 2, 0)));
    SA_EXPECT_OK(step->run_micro_batch(rows_from(2, 2, 1)));
    SA_EXPECT_OK(step->finish(results.bound));

    expect_results_mean_of(results, first.results, second.results);
}
