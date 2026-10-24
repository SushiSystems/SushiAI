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
        rows.rows = RowRange{first, count};
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
} // namespace

/** @brief Tests that make() refuses a micro-batch of no row before it creates a runtime. */
TEST(Small_ReplicaStep, MakeRefusesAMicroBatchBelowOne)
{
    SA_EXPECT_FAILURE(ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN}, 0),
                      StatusCode::InvalidDescription, "holds 0 rows");
}

/** @brief Tests that a step that ran no micro-batch writes zeros. */
TEST(Medium_ReplicaStep, NoMicroBatchWritesZeros)
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

    for (std::size_t index = 0; index < step->result_layout().entries().size(); ++index)
    {
        const std::vector<unsigned char> bytes = results.entry(index);
        EXPECT_EQ(bytes, std::vector<unsigned char>(bytes.size(), 0)) << "result " << index;
    }
}

/** @brief Tests that one micro-batch writes the replay's results byte for byte. */
TEST(Medium_ReplicaStep, OneMicroBatchWritesTheReplayResultsBitForBit)
{
    const SingleMachineStep reference(4);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ReplicaStep> step,
                            ReplicaStep::make(linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN},
                                              4));
    HostPack parameters(step->parameter_layout());
    HostPack results(step->result_layout());
    fill_parameters(step->parameter_layout(), parameters);

    SA_EXPECT_OK(step->begin(parameters.bound));
    SA_EXPECT_OK(step->run_micro_batch(rows_from(0, 4, 0)));
    SA_EXPECT_OK(step->finish(results.bound));

    ASSERT_EQ(step->result_layout().entries().size(), reference.results.size());
    for (std::size_t index = 0; index < reference.results.size(); ++index)
        EXPECT_EQ(results.entry(index), reference.results[index]) << "result " << index;
}

/**
 * @brief Tests that two micro-batches write (a + b) / 2 of the two replays, in float32 and in
 *        micro-batch order, bit for bit.
 */
TEST(Medium_ReplicaStep, TwoMicroBatchesWriteTheirMeanInMicroBatchOrder)
{
    const SingleMachineStep first(2, 0);
    const SingleMachineStep second(2, 2);
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

    for (std::size_t index = 0; index < first.results.size(); ++index)
    {
        const std::vector<float> got = as_floats(results.entry(index));
        const std::vector<float> a = as_floats(first.results[index]);
        const std::vector<float> b = as_floats(second.results[index]);
        ASSERT_EQ(got.size(), a.size()) << "result " << index;
        for (std::size_t element = 0; element < got.size(); ++element)
        {
            const float expected = (a[element] + b[element]) / 2.0f;
            EXPECT_EQ(std::memcmp(&got[element], &expected, sizeof(float)), 0)
                << "result " << index << ", element " << element;
        }
    }
}
