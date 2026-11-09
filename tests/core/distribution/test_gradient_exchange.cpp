/**************************************************************************/
/* test_gradient_exchange.cpp                                             */
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
 * @file test_gradient_exchange.cpp
 * @brief Tests GradientExchange: two replicas against one machine, two pinned runs, every type.
 * @author Mustafa Garip
 */

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <SushiBLAS/cluster/bound_pack.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/remote_task.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/api/graph/node_options.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/distributed/buffer_view.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/gradient_exchange.hpp>
#include <SushiAI/distribution/parameter_publish.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/execution/runtime_execution_backend.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/shape/shape.hpp>

#include "exchange_test_support.hpp"
#include "replica_test_support.hpp"
#include <support/tolerance.hpp>
#include <support/test_names.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    using namespace SushiAI::Distribution::ExchangeSupport;
    using namespace SushiAI::Distribution::Support;
    namespace Distributed = SushiRuntime::Distributed;

    /**
     * @brief Bounds the absolute difference of a float32 sum against one machine's step; 32 unit
     * roundoffs of float, the smallest power of two that covers 1e-6.
     */
    constexpr double K_ABSOLUTE_TOLERANCE = 32.0 * TestSupport::unit_roundoff<float>();

    /**
     * @brief Bounds the relative difference of a float32 sum against one machine's step; 256 unit
     * roundoffs of float, the smallest power of two that covers 1e-5.
     */
    constexpr double K_RELATIVE_TOLERANCE = 256.0 * TestSupport::unit_roundoff<float>();

    /** @brief Holds the rows of one batch: two replicas of two rows. */
    constexpr std::int64_t K_BATCH_ROWS = 4;

    /** @brief Holds the batches of one epoch the exchange counts in. */
    constexpr std::uint64_t K_BATCHES_PER_EPOCH = 8;

    /** @brief Holds the elements of each tensor of the seven-type pack. */
    constexpr std::int64_t K_TYPED_ELEMENTS = 3;

    /** @brief Holds the bytes of each target, in result pack order. */
    using ResultBytes = std::vector<std::vector<unsigned char>>;

    /** @brief Returns the bytes of each tensor of @p tensors. */
    ResultBytes bytes_of(SushiRuntime::span<const SushiBLAS::Tensor> tensors)
    {
        ResultBytes bytes;
        for (const SushiBLAS::Tensor& tensor : tensors)
        {
            const std::size_t size = static_cast<std::size_t>(tensor.num_elements)
                                     * dtype_size(tensor.dtype);
            const unsigned char* const first = static_cast<const unsigned char*>(tensor.data());
            bytes.emplace_back(first, first + size);
        }
        return bytes;
    }

    /** @brief Returns the float32 elements of a result's bytes. */
    std::vector<float> as_floats(const std::vector<unsigned char>& bytes)
    {
        std::vector<float> values(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
        return values;
    }

    /** @brief Expects each float32 result within the tolerances of its reference. */
    void expect_near(const ResultBytes& got, const ResultBytes& expected)
    {
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t index = 0; index < got.size(); ++index)
        {
            const std::vector<float> values = as_floats(got[index]);
            const std::vector<float> reference = as_floats(expected[index]);
            ASSERT_EQ(values.size(), reference.size()) << "result " << index;
            for (std::size_t element = 0; element < values.size(); ++element)
            {
                const double bound = K_ABSOLUTE_TOLERANCE
                                     + K_RELATIVE_TOLERANCE * std::fabs(reference[element]);
                EXPECT_NEAR(values[element], reference[element], bound)
                    << "result " << index << ", element " << element;
            }
        }
    }

    /**
     * @brief Runs @p replays replays of an exchange of two LinearModel replicas, replica 0 on
     *        the master and replica 1 on the worker, and returns the targets after each.
     *
     * The step holds no optimizer, so every replay reads the parameters of the first.
     */
    std::vector<ResultBytes> exchange_replays(std::size_t replays)
    {
        const ClusterSection section = two_replica_section();
        std::unique_ptr<ReplicaKernel> master_kernel =
            ReplicaKernel::make(section.plan(), linear_factory(), ReplicaDraws{K_SEED, 0, K_RUN})
                .value();
        std::unique_ptr<ReplicaKernel> worker_kernel =
            ReplicaKernel::make(section.plan(), linear_factory(), ReplicaDraws{K_SEED, 1, K_RUN},
                                worker_replica_config())
                .value();
        TwoMachines machines(
            section,
            [&](Distributed::KernelRegistry& kernels) { master_kernel->register_with(kernels); },
            [&](Distributed::KernelRegistry& kernels) { worker_kernel->register_with(kernels); });

        SushiBLAS::Engine engine(machines.master_runtime());
        SushiAI::RuntimeExecutionBackend backend(machines.master_runtime(), engine);
        std::unique_ptr<ReplicaModel> built = LinearModel::make(engine, K_BATCH_ROWS).value();
        LinearModel& model = static_cast<LinearModel&>(*built);
        model.fill_parameters();

        const SushiRuntime::span<const SushiBLAS::Tensor> parameters = model.parameters();
        const SushiRuntime::span<const SushiBLAS::Tensor> results = model.results();
        std::vector<SushiBLAS::Tensor> published(parameters.begin(), parameters.end());
        std::vector<SushiBLAS::Tensor> targets(results.begin(), results.end());
        std::unique_ptr<ParameterPublish> publish =
            ParameterPublish::make(backend, machines.binding(), master_kernel->parameter_layout(),
                                   std::move(published)).value();

        ExchangeRun run;
        run.operation = master_kernel->operation();
        run.seed = K_SEED;
        run.run_index = K_RUN;
        run.batches_per_epoch = K_BATCHES_PER_EPOCH;
        std::unique_ptr<GradientExchange> exchange =
            GradientExchange::make(backend, machines.binding(), section.plan(), run,
                                   publish->pack(), master_kernel->result_layout(),
                                   std::move(targets)).value();

        Graph::StepRecorderList trailing;
        SA_EXPECT_OK(trailing.add(*exchange));
        SA_EXPECT_OK(trailing.add(*publish));
        const Graph::Graph empty;
        std::unique_ptr<Graph::CompiledStep> step =
            Graph::CompiledStep::compile(backend, empty, model.pool(), &trailing).value();
        // The position gather, two replicas, one float32 segment, the scatter, the publish.
        EXPECT_EQ(step->operation_count(), 6u);

        std::vector<ResultBytes> seen;
        for (std::size_t replay = 0; replay < replays; ++replay)
        {
            SA_EXPECT_OK(step->run());
            seen.push_back(bytes_of(results));
        }
        EXPECT_EQ(machines.cluster().offloaded(), replays)
            << "replica 1 ran somewhere other than the worker";
        return seen;
    }

    /** @brief Returns element @p index of replica @p replica's result, for a real type. */
    template<typename Element>
    Element real_value(std::uint64_t replica, std::size_t index)
    {
        return static_cast<Element>(static_cast<float>(replica + 1) + 0.25f * index);
    }

    /** @brief Returns element @p index of replica @p replica's result, for a complex type. */
    template<typename Component>
    std::complex<Component> complex_value(std::uint64_t replica, std::size_t index)
    {
        return {real_value<Component>(replica, index),
                static_cast<Component>(-static_cast<Component>(replica + 1))};
    }

    /** @brief Returns element @p index of replica @p replica's result, for an integer type. */
    template<typename Element>
    Element integer_value(std::uint64_t replica, std::size_t index)
    {
        return static_cast<Element>((replica + 1) * 10 + index);
    }

    /** @brief Writes replica @p replica's values into a span of one type. */
    void write_values(const SushiBLAS::PackSpan& span, std::uint64_t replica)
    {
        for (std::size_t index = 0; index < span.count; ++index)
        {
            switch (span.dtype)
            {
                case DType::Half:
                    span.as<sycl::half>()[index] = real_value<sycl::half>(replica, index);
                    break;
                case DType::Float32:
                    span.as<float>()[index] = real_value<float>(replica, index);
                    break;
                case DType::Float64:
                    span.as<double>()[index] = real_value<double>(replica, index);
                    break;
                case DType::Complex32:
                    span.as<std::complex<float>>()[index] = complex_value<float>(replica, index);
                    break;
                case DType::Complex64:
                    span.as<std::complex<double>>()[index] =
                        complex_value<double>(replica, index);
                    break;
                case DType::Int32:
                    span.as<std::int32_t>()[index] = integer_value<std::int32_t>(replica, index);
                    break;
                case DType::Int64:
                    span.as<std::int64_t>()[index] = integer_value<std::int64_t>(replica, index);
                    break;
                case DType::UInt32:
                    span.as<std::uint32_t>()[index] =
                        integer_value<std::uint32_t>(replica, index);
                    break;
            }
        }
    }

    /** @brief Returns the bytes of a tensor of @p dtype holding the sum of both replicas. */
    std::vector<unsigned char> expected_sum(DType dtype)
    {
        std::vector<unsigned char> bytes(static_cast<std::size_t>(K_TYPED_ELEMENTS)
                                         * dtype_size(dtype));
        for (std::size_t index = 0; index < static_cast<std::size_t>(K_TYPED_ELEMENTS); ++index)
        {
            unsigned char* const place = bytes.data() + index * dtype_size(dtype);
            // Every value is a multiple of 1/8, so each weighted sum is exact in every type.
            switch (dtype)
            {
                case DType::Half:
                {
                    const float first = static_cast<float>(real_value<sycl::half>(0, index));
                    const float second = static_cast<float>(real_value<sycl::half>(1, index));
                    const sycl::half value(0.5f * first + 0.5f * second);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Float32:
                {
                    const float value = 0.5f * real_value<float>(0, index)
                                        + 0.5f * real_value<float>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Float64:
                {
                    const double value = 0.5 * real_value<double>(0, index)
                                         + 0.5 * real_value<double>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Complex32:
                {
                    const std::complex<float> value = 0.5f * complex_value<float>(0, index)
                                                      + 0.5f * complex_value<float>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Complex64:
                {
                    const std::complex<double> value = 0.5 * complex_value<double>(0, index)
                                                       + 0.5 * complex_value<double>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Int32:
                {
                    const std::int32_t value = integer_value<std::int32_t>(0, index)
                                               + integer_value<std::int32_t>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::Int64:
                {
                    const std::int64_t value = integer_value<std::int64_t>(0, index)
                                               + integer_value<std::int64_t>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
                case DType::UInt32:
                {
                    const std::uint32_t value = integer_value<std::uint32_t>(0, index)
                                                + integer_value<std::uint32_t>(1, index);
                    std::memcpy(place, &value, sizeof(value));
                    break;
                }
            }
        }
        return bytes;
    }

    /**
     * @brief Registers under @p operation a kernel that writes the replica's values into every
     *        tensor of a result pack of @p layout.
     */
    void register_typed_kernel(Distributed::KernelRegistry& kernels,
                               SushiRuntime::OperationID operation,
                               const SushiBLAS::PackLayout& layout)
    {
        kernels.register_kernel(operation,
                                [layout](sycl::queue&,
                                         SushiRuntime::span<const Distributed::BufferView>,
                                         SushiRuntime::span<const Distributed::BufferView> outputs,
                                         const SushiRuntime::NodeOptions& options) -> sycl::event
        {
            const ReplicaTask task = ReplicaTask::decode(options.parameters).value();
            const std::vector<SushiBLAS::BoundPack> bound = SushiBLAS::bind_views(
                outputs, SushiRuntime::span<const SushiBLAS::PackLayout>(&layout, 1));
            for (std::size_t entry = 0; entry < layout.entries().size(); ++entry)
                write_values(bound.front().tensor(entry), task.replica_index);
            return sycl::event{};
        });
    }
    /** @brief Expects each replay of @p first to equal the same replay of @p second byte for byte. */
    void expect_same_replays(const std::vector<ResultBytes>& first,
                             const std::vector<ResultBytes>& second)
    {
        ASSERT_EQ(first.size(), second.size());
        for (std::size_t replay = 0; replay < first.size(); ++replay)
            EXPECT_EQ(first[replay], second[replay]) << "replay " << replay;
    }

    /** @brief Lists the types ordered_sum() sums; UInt32, the eighth, is refused by make(). */
    const std::vector<DType>& typed_dtypes()
    {
        static const std::vector<DType> types = {DType::Half,      DType::Float32, DType::Float64,
                                                 DType::Complex32, DType::Complex64, DType::Int32,
                                                 DType::Int64};
        return types;
    }

    /** @brief Holds what one exchange of a pack of every summed type left behind. */
    struct TypedExchange
    {
        /** @brief Holds whether the step ran. */
        bool ran = false;

        /** @brief Holds how many tasks the master offloaded to the worker. */
        std::size_t offloaded = 0;

        /** @brief Holds the bytes of each target after the exchange. */
        ResultBytes targets;
    };

    /**
     * @brief Runs one exchange of a result pack holding each type ordered_sum() sums, and returns
     *        what it left behind.
     */
    TypedExchange run_typed_exchange()
    {
        const ClusterSection section = two_replica_section();
        const std::vector<DType>& types = typed_dtypes();

        SushiRuntime::Runtime layout_runtime = named_runtime(K_MASTER_NAME);
        SushiBLAS::Engine layout_engine(layout_runtime);
        std::vector<SushiBLAS::Tensor> shapes;
        for (const DType dtype : types)
            shapes.push_back(
                allocate_tensor(layout_engine, Shape{K_TYPED_ELEMENTS}, dtype).value());
        const SushiBLAS::PackLayout layout = SushiBLAS::PackLayout::of(
            SushiRuntime::span<const SushiBLAS::Tensor>(shapes.data(), shapes.size()));
        const SushiRuntime::OperationID operation =
            SushiBLAS::remote_operation("sushiai.test.typed", layout.fingerprint());

        const auto setup = [&](Distributed::KernelRegistry& kernels)
        {
            register_typed_kernel(kernels, operation, layout);
        };
        TwoMachines machines(section, setup, setup);
        SushiBLAS::Engine engine(machines.master_runtime());
        SushiAI::RuntimeExecutionBackend backend(machines.master_runtime(), engine);

        std::vector<SushiBLAS::Tensor> targets;
        for (const DType dtype : types)
            targets.push_back(allocate_tensor(engine, Shape{K_TYPED_ELEMENTS}, dtype).value());
        const SushiBLAS::Tensor parameter =
            allocate_tensor(engine, Shape{1}, DType::Float32).value();
        const SushiBLAS::PackLayout parameter_layout =
            SushiBLAS::PackLayout::of(SushiRuntime::span<const SushiBLAS::Tensor>(&parameter, 1));
        std::unique_ptr<ParameterPublish> publish =
            ParameterPublish::make(backend, machines.binding(), parameter_layout, {parameter})
                .value();

        ExchangeRun run;
        run.operation = operation;
        run.seed = K_SEED;
        run.run_index = K_RUN;
        std::unique_ptr<GradientExchange> exchange =
            GradientExchange::make(backend, machines.binding(), section.plan(), run,
                                   publish->pack(), layout, targets)
                .value();

        const Graph::Graph empty;
        Graph::TensorPool pool(engine);
        std::unique_ptr<Graph::CompiledStep> step =
            Graph::CompiledStep::compile(backend, empty, pool, exchange.get()).value();

        TypedExchange exchanged;
        exchanged.ran = step->run().is_ok();
        exchanged.offloaded = machines.cluster().offloaded();
        exchanged.targets = bytes_of(
            SushiRuntime::span<const SushiBLAS::Tensor>(targets.data(), targets.size()));
        return exchanged;
    }

    /** @brief Returns the exchange of the typed pack, run once for every case that reads it. */
    const TypedExchange& typed_exchange()
    {
        static const TypedExchange exchanged = run_typed_exchange();
        return exchanged;
    }

    /** @brief Runs each type the exchange sums. */
    class Medium_GradientExchangeType : public ::testing::TestWithParam<std::size_t>
    {

    };
} // namespace

namespace
{
    /** @brief Gives each case the one backend of the program. */
    class Medium_GradientExchange : public SushiAI::Test::EngineTest {};
} // namespace

/**
 * @brief Tests that two replicas of two rows, summed with weights 1/2, give the gradients and
 *        the loss of one step over the four rows within float32 tolerance.
 *
 * The tolerance is the one of a replica of two micro-batches: each replica's matrix product
 * sums over two rows instead of four, which moves each result by a few float32 ulps.
 */
TEST_F(Medium_GradientExchange, TwoReplicasSumToOneStepOverTheWholeBatchWithinTolerance)
{
    const SingleMachineStep reference(backend(), K_BATCH_ROWS);
    const std::vector<ResultBytes> seen = exchange_replays(1);

    ASSERT_EQ(seen.size(), 1u);
    expect_near(seen.front(), reference.results);
}

/** @brief Tests that the second replay serves the second batch of the epoch. */
TEST_F(Medium_GradientExchange, TheSecondReplayServesTheNextBatch)
{
    const SingleMachineStep reference(backend(), K_BATCH_ROWS,
                                      static_cast<std::uint64_t>(K_BATCH_ROWS));
    const std::vector<ResultBytes> seen = exchange_replays(2);

    ASSERT_EQ(seen.size(), 2u);
    expect_near(seen.back(), reference.results);
}

/** @brief Tests that two runs with the same rules and seed give the same bytes. */
TEST_F(Medium_GradientExchange, TwoPinnedRunsEqualEachOtherBitForBit)
{
    const std::vector<ResultBytes> first = exchange_replays(2);
    const std::vector<ResultBytes> second = exchange_replays(2);

    expect_same_replays(first, second);
}

/** @brief Tests that a result pack holding each summed type crosses the fabric to the worker. */
TEST_F(Medium_GradientExchange, ATypedPackCrossesToTheWorker)
{
    const TypedExchange& exchanged = typed_exchange();

    EXPECT_TRUE(exchanged.ran);
    EXPECT_EQ(exchanged.offloaded, 1u);
}

/**
 * @brief Tests that a type sums in replica order: weighted for the floating and complex types,
 *        exactly for Int32 and Int64. UInt32, the eighth, is refused by make(); the case
 *        MakeRefusesAUInt32Result pins it.
 */
TEST_P(Medium_GradientExchangeType, ATypeSumsInReplicaOrder)
{
    const DType dtype = typed_dtypes()[GetParam()];
    const TypedExchange& exchanged = typed_exchange();
    ASSERT_EQ(exchanged.targets.size(), typed_dtypes().size());

    EXPECT_EQ(exchanged.targets[GetParam()], expected_sum(dtype));
}

INSTANTIATE_TEST_SUITE_P(Types, Medium_GradientExchangeType,
                         ::testing::Range<std::size_t>(0, 7),
                         [](const ::testing::TestParamInfo<std::size_t>& info)
                         {
                             return ::SushiAI::Test::parameter_name(dtype_name(typed_dtypes()[info.param]));
                         });
