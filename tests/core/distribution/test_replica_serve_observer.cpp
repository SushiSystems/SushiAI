/**************************************************************************/
/* test_replica_serve_observer.cpp                                        */
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
 * @file test_replica_serve_observer.cpp
 * @brief Tests that a replica kernel tells its observer of each task, with its position.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_serve_observer.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Records the replica and the step of each task it hears of. */
    class RecordingObserver final : public ReplicaServeObserver
    {
        public:
            void on_served(std::uint64_t replica, const ReplicaPosition& position) override
            {
                replicas.push_back(replica);
                steps.push_back(position.step);
            }

            /// @brief The replica of each task, in order.
            std::vector<std::uint64_t> replicas;

            /// @brief The step of each task, in order.
            std::vector<std::uint64_t> steps;
    };

    /** @brief Serves replica 0 of @p kernel the step @p step of a one-batch epoch. */
    Status serve_step(ReplicaKernel& kernel, std::uint64_t step)
    {
        std::vector<unsigned char> parameters(kernel.parameter_layout().size_bytes());
        std::vector<unsigned char> results(kernel.result_layout().size_bytes());
        const ReplicaPosition position = ReplicaPosition::at_step(step, 1);
        std::uint64_t words[3] = {position.epoch, position.batch_index, position.step};
        const Distributed::BufferView inputs[2] = {{parameters.data(), parameters.size()},
                                                   {words, sizeof(words)}};
        const Distributed::BufferView outputs[1] = {{results.data(), results.size()}};

        ReplicaTask task;
        task.replica_count = 1;
        task.run_index = K_RUN;
        task.seed = K_SEED;
        const auto encoded = task.encode();
        SushiRuntime::NodeOptions options;
        options.parameters =
            SushiRuntime::span<const std::uint64_t>(encoded.data(), encoded.size());
        return kernel.serve(SushiRuntime::span<const Distributed::BufferView>(inputs, 2),
                            SushiRuntime::span<const Distributed::BufferView>(outputs, 1),
                            options);
    }
} // namespace

/** @brief Tests that the observer hears of each task in order, and nothing once removed. */
TEST(Medium_ReplicaServeObserver, TheKernelTellsTheObserverEachTaskAndItsStep)
{
    ReplicaPlan plan = ReplicaPlan::make({ReplicaShare{4, 4}}).value();
    const std::unique_ptr<ReplicaKernel> kernel =
        ReplicaKernel::make(std::move(plan), linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN})
            .value();
    RecordingObserver observer;
    kernel->observe(&observer);

    SA_EXPECT_OK(serve_step(*kernel, 2));
    SA_EXPECT_OK(serve_step(*kernel, 5));
    kernel->observe(nullptr);
    SA_EXPECT_OK(serve_step(*kernel, 6));

    EXPECT_EQ(observer.replicas, (std::vector<std::uint64_t>{0, 0}));
    EXPECT_EQ(observer.steps, (std::vector<std::uint64_t>{2, 5}));
}
