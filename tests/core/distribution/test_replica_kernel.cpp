/**************************************************************************/
/* test_replica_kernel.cpp                                                */
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
 * @file test_replica_kernel.cpp
 * @brief Tests that one replica served on InProcessFabric returns a single-machine step's results.
 * @author Mustafa Garip
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/api/graph/recorder.hpp>
#include <SushiRuntime/cluster_load/offload_policy.hpp>
#include <SushiRuntime/core/placement_class.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiRuntime/distributed/worker_agent.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>

#include "replica_test_support.hpp"

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::Support;
    namespace Distributed = SushiRuntime::Distributed;

    /** @brief Holds how long a case waits for the worker and for the run. */
    constexpr std::chrono::seconds K_WAIT{10};

    /** @brief Names the device the packs live on: the engine's default, which its tensors use. */
    constexpr std::size_t K_DEVICE = 0;

    /** @brief Bounds the absolute difference of a float32 result over two micro-batches. */
    constexpr double K_ABSOLUTE_TOLERANCE = 1e-6;

    /** @brief Bounds the relative difference of a float32 result over two micro-batches. */
    constexpr double K_RELATIVE_TOLERANCE = 1e-5;

    /** @brief Returns options whose policy sends a task of any size to a worker. */
    Distributed::ClusterOptions eager_options()
    {
        Distributed::ClusterOptions options;
        options.policy = std::make_shared<Distributed::LeastLoadedOffloadPolicy>(0);
        return options;
    }

    /** @brief Returns a registry whose kernel for @p operation fails, so a local run shows. */
    SushiRuntime::sushi_ptr<Distributed::KernelRegistry> refusing_kernels(
        SushiRuntime::OperationID operation)
    {
        SushiRuntime::sushi_ptr<Distributed::KernelRegistry> kernels =
            SushiRuntime::make_sushi<Distributed::KernelRegistry>();
        kernels->register_kernel(operation,
                                 [](sycl::queue&,
                                    SushiRuntime::span<const Distributed::BufferView>,
                                    SushiRuntime::span<const Distributed::BufferView>,
                                    const SushiRuntime::NodeOptions&) -> sycl::event
        {
            throw std::runtime_error("the replica task ran on the master");
        });
        return kernels;
    }

    /**
     * @brief Serves replica 0 of a one-replica plan on a worker over InProcessFabric, and
     *        returns the bytes of each entry of the result pack the master received.
     * @param rows The replica's rows per step, r_0.
     * @param micro_batch Its micro-batch, m_0.
     * @param batch_index The batch the position pack names, of epoch 0.
     */
    std::vector<std::vector<unsigned char>> serve_one_replica(std::int64_t rows,
                                                              std::int64_t micro_batch,
                                                              std::uint64_t batch_index = 0)
    {
        std::vector<std::vector<unsigned char>> received;
        ReplicaPlan plan = ReplicaPlan::make({ReplicaShare{rows, micro_batch}}).value();
        std::unique_ptr<ReplicaKernel> kernel =
            ReplicaKernel::make(std::move(plan), linear_factory(),
                                ReplicaDraws{K_SEED, 0, K_RUN}).value();
        EXPECT_EQ(kernel->operation_name().rfind("sushiai.replica:", 0), 0u);
        EXPECT_EQ(kernel->operation_name().size(), std::string("sushiai.replica:").size() + 16);

        SushiRuntime::sushi_ptr<Distributed::KernelRegistry> worker_kernels =
            SushiRuntime::make_sushi<Distributed::KernelRegistry>();
        kernel->register_with(*worker_kernels);

        Distributed::InProcessFabric fabric;
        SushiRuntime::Runtime worker_runtime = SushiRuntime::Runtime::create().value();
        Distributed::WorkerAgent agent(worker_runtime,
                                       Distributed::InProcessTransport::make_worker(fabric),
                                       worker_kernels);

        SushiRuntime::Runtime master_runtime = SushiRuntime::Runtime::create().value();
        Distributed::Cluster cluster =
            Distributed::Cluster::attach(master_runtime,
                                         Distributed::InProcessTransport::make_master(fabric),
                                         eager_options()).value();
        EXPECT_TRUE(agent.start().is_ok());
        EXPECT_TRUE(cluster.wait_for_workers(1, K_WAIT));

        SushiBLAS::ClusterBinding binding(cluster, refusing_kernels(kernel->operation()));
        SushiBLAS::TensorPack parameters(master_runtime, kernel->parameter_layout(),
                                         K_DEVICE);
        SushiBLAS::TensorPack results(master_runtime, kernel->result_layout(), K_DEVICE);
        binding.share(parameters);
        binding.share(results);

        SushiBLAS::Engine engine(master_runtime);
        const SushiBLAS::Tensor position_words =
            allocate_tensor(engine, Shape{2}, DType::Int64).value();
        SushiBLAS::TensorPack position(
            master_runtime,
            SushiBLAS::PackLayout::of(
                SushiRuntime::span<const SushiBLAS::Tensor>(&position_words, 1)),
            position_words.storage->device_index());
        binding.share(position);
        const std::uint64_t named[2] = {0, batch_index};
        std::memcpy(position.data(), named, K_REPLICA_POSITION_BYTES);

        std::size_t element = 0;
        for (const SushiBLAS::PackEntry& entry : parameters.layout().entries())
        {
            float* const values = reinterpret_cast<float*>(
                static_cast<unsigned char*>(parameters.data()) + entry.offset_bytes);
            for (std::size_t index = 0; index < entry.element_count; ++index)
                values[index] = parameter_value(element++);
        }

        ReplicaTask task;
        task.replica_index = 0;
        task.replica_count = 1;
        task.run_index = K_RUN;
        task.seed = K_SEED;
        const auto words = task.encode();
        SushiRuntime::NodeOptions options;
        options.name = "replica 0";
        options.operation = kernel->operation();
        options.parameters = SushiRuntime::span<const std::uint64_t>(words.data(), words.size());
        options.placement_class = SushiRuntime::PlacementClass(K_REPLICA_PLACEMENT_CLASS.data(),
                                                               K_REPLICA_PLACEMENT_CLASS.size());
        options.placement_key = 0;

        const SushiBLAS::TensorPack* const inputs[2] = {&parameters, &position};
        const SushiBLAS::TensorPack* const outputs[1] = {&results};
        SushiRuntime::Graph graph = master_runtime.graph();
        graph.extend([&](SushiRuntime::Recorder& recorder)
        {
            SushiBLAS::record_remote(
                recorder, binding, options,
                SushiRuntime::span<const SushiBLAS::TensorPack* const>(inputs, 2),
                SushiRuntime::span<const SushiBLAS::TensorPack* const>(outputs, 1));
        });
        EXPECT_TRUE(graph.run().has_value());
        EXPECT_EQ(cluster.offloaded(), 1u) << "the replica task ran on the master";
        agent.stop();

        for (const SushiBLAS::PackEntry& entry : results.layout().entries())
        {
            const unsigned char* const first =
                static_cast<const unsigned char*>(results.data()) + entry.offset_bytes;
            received.emplace_back(first, first + entry.size_bytes);
        }
        return received;
    }

    /** @brief Returns the float32 elements of a result's bytes. */
    std::vector<float> as_floats(const std::vector<unsigned char>& bytes)
    {
        std::vector<float> values(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        return values;
    }
} // namespace

/**
 * @brief Tests that one replica of one micro-batch, served by a worker, returns the gradients
 *        and the loss of the single-machine step over the same rows, bit for bit.
 */
TEST(Medium_ReplicaKernel, OneMicroBatchReturnsTheSingleMachineResultsBitForBit)
{
    constexpr std::int64_t K_ROWS = 4;
    const SingleMachineStep reference(K_ROWS);
    const std::vector<std::vector<unsigned char>> received = serve_one_replica(K_ROWS, K_ROWS);

    ASSERT_EQ(received.size(), reference.results.size());
    for (std::size_t index = 0; index < received.size(); ++index)
        EXPECT_EQ(received[index], reference.results[index]) << "result " << index;
}

/**
 * @brief Tests that two micro-batches of two rows return the mean the single-machine step
 *        over four rows computes, within float32 tolerance, since the sum over rows is split.
 */
TEST(Medium_ReplicaKernel, TwoMicroBatchesReturnTheSingleMachineMeanWithinTolerance)
{
    const SingleMachineStep reference(4);
    const std::vector<std::vector<unsigned char>> received = serve_one_replica(4, 2);

    ASSERT_EQ(received.size(), reference.results.size());
    for (std::size_t index = 0; index < received.size(); ++index)
    {
        const std::vector<float> got = as_floats(received[index]);
        const std::vector<float> expected = as_floats(reference.results[index]);
        ASSERT_EQ(got.size(), expected.size()) << "result " << index;
        for (std::size_t element = 0; element < got.size(); ++element)
        {
            const double bound = K_ABSOLUTE_TOLERANCE
                                 + K_RELATIVE_TOLERANCE * std::fabs(expected[element]);
            EXPECT_NEAR(got[element], expected[element], bound)
                << "result " << index << ", element " << element;
        }
    }
}

/** @brief Tests that the batch the position pack names picks the rows the replica runs. */
TEST(Medium_ReplicaKernel, ThePositionPackPicksTheBatch)
{
    constexpr std::int64_t K_ROWS = 4;
    const SingleMachineStep reference(K_ROWS, static_cast<std::uint64_t>(K_ROWS));
    const std::vector<std::vector<unsigned char>> received = serve_one_replica(K_ROWS, K_ROWS, 1);

    ASSERT_EQ(received.size(), reference.results.size());
    for (std::size_t index = 0; index < received.size(); ++index)
        EXPECT_EQ(received[index], reference.results[index]) << "result " << index;
}

/** @brief Tests that a node without its position pack is refused before any step runs. */
TEST(Medium_ReplicaKernel, ANodeWithoutItsPositionPackIsRefused)
{
    ReplicaPlan plan = ReplicaPlan::make({ReplicaShare{4, 4}}).value();
    std::unique_ptr<ReplicaKernel> kernel =
        ReplicaKernel::make(std::move(plan), linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN})
            .value();

    std::vector<unsigned char> parameters(kernel->parameter_layout().size_bytes());
    std::vector<unsigned char> results(kernel->result_layout().size_bytes());
    const Distributed::BufferView inputs[1] = {{parameters.data(), parameters.size()}};
    const Distributed::BufferView outputs[1] = {{results.data(), results.size()}};

    ReplicaTask task;
    task.run_index = K_RUN;
    task.seed = K_SEED;
    const auto words = task.encode();
    SushiRuntime::NodeOptions options;
    options.parameters = SushiRuntime::span<const std::uint64_t>(words.data(), words.size());

    SA_EXPECT_FAILURE(kernel->serve(SushiRuntime::span<const Distributed::BufferView>(inputs, 1),
                                    SushiRuntime::span<const Distributed::BufferView>(outputs, 1),
                                    options),
                      StatusCode::InvalidDescription, "1 inputs");
}
