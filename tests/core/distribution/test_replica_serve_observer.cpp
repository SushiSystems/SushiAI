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
#include <SushiAI/distribution/in_memory_replica_serve_observer.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Returns the replica of each task @p observer heard of, in order. */
    std::vector<std::uint64_t> replicas_heard_by(const InMemoryReplicaServeObserver& observer)
    {
        std::vector<std::uint64_t> replicas;
        for (const InMemoryReplicaServeObserver::Served& task : observer.served())
            replicas.push_back(task.replica);
        return replicas;
    }

    /** @brief Returns the step of each task @p observer heard of, in order. */
    std::vector<std::uint64_t> steps_heard_by(const InMemoryReplicaServeObserver& observer)
    {
        std::vector<std::uint64_t> steps;
        for (const InMemoryReplicaServeObserver::Served& task : observer.served())
            steps.push_back(task.position.step);
        return steps;
    }

    /** @brief Returns a kernel of one replica of four rows in one micro-batch. */
    std::unique_ptr<ReplicaKernel> make_kernel()
    {
        ReplicaPlan plan = ReplicaPlan::make({ReplicaShare{4, 4}}).value();
        return ReplicaKernel::make(std::move(plan), linear_factory(),
                                   ReplicaDraws{K_SEED, 0, K_RUN}).value();
    }

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

/** @brief Tests that the observer hears of the replica of each task, in order. */
TEST(Medium_ReplicaServeObserver, TheKernelTellsTheObserverTheReplicaOfEachTask)
{
    const std::unique_ptr<ReplicaKernel> kernel = make_kernel();
    InMemoryReplicaServeObserver observer;
    kernel->observe(&observer);

    SA_EXPECT_OK(serve_step(*kernel, 2));
    SA_EXPECT_OK(serve_step(*kernel, 5));

    EXPECT_EQ(replicas_heard_by(observer), (std::vector<std::uint64_t>{0, 0}));
}

/** @brief Tests that the observer hears of the step of each task, in order. */
TEST(Medium_ReplicaServeObserver, TheKernelTellsTheObserverTheStepOfEachTask)
{
    const std::unique_ptr<ReplicaKernel> kernel = make_kernel();
    InMemoryReplicaServeObserver observer;
    kernel->observe(&observer);

    SA_EXPECT_OK(serve_step(*kernel, 2));
    SA_EXPECT_OK(serve_step(*kernel, 5));

    EXPECT_EQ(steps_heard_by(observer), (std::vector<std::uint64_t>{2, 5}));
}

/** @brief Tests that an observer removed from the kernel hears of no further task. */
TEST(Medium_ReplicaServeObserver, ARemovedObserverHearsOfNoFurtherTask)
{
    const std::unique_ptr<ReplicaKernel> kernel = make_kernel();
    InMemoryReplicaServeObserver observer;
    kernel->observe(&observer);
    SA_EXPECT_OK(serve_step(*kernel, 2));

    kernel->observe(nullptr);
    SA_EXPECT_OK(serve_step(*kernel, 6));

    EXPECT_EQ(observer.served_count(), 1u);
}
