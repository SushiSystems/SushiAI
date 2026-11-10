/**************************************************************************/
/* test_parameter_publish.cpp                                             */
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
 * @file test_parameter_publish.cpp
 * @brief Tests ParameterPublish: the first copy, the gather of each replay and its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/core/sushi_ptr.hpp>
#include <SushiRuntime/distributed/cluster.hpp>
#include <SushiRuntime/in_process_transport/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/parameter_publish.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>
#include <support/test_names.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    namespace Distributed = SushiRuntime::Distributed;

    /**
     * @brief Holds a master with no worker, attached to the runtime of the program, its binding,
     *        and two float32 parameters of 3 and 2 elements.
     */
    class Medium_ParameterPublish : public SushiAI::Test::EngineTest
    {
        protected:
            /** @brief Attaches the master to the program's runtime and allocates the parameters. */
            void SetUp() override
            {
                SushiAI::Test::EngineTest::SetUp();
                cluster_ = std::make_unique<Distributed::Cluster>(
                    Distributed::Cluster::attach(
                        SushiAI::Test::TestEnvironment::runtime(),
                        Distributed::InProcessTransport::make_master(fabric_))
                        .value());
                binding_ = std::make_unique<SushiBLAS::ClusterBinding>(
                    *cluster_, SushiRuntime::make_sushi<Distributed::KernelRegistry>());
                parameters_.push_back(allocate_tensor(*engine, Shape{3}, DType::Float32).value());
                parameters_.push_back(allocate_tensor(*engine, Shape{2}, DType::Float32).value());
                fill(1.0f);
            }

            /** @brief Releases the parameters and the binding, then detaches the master, so the
             *         next test attaches its own to the same runtime. */
            void TearDown() override
            {
                parameters_.clear();
                binding_.reset();
                cluster_->detach();
                cluster_.reset();
                SushiAI::Test::EngineTest::TearDown();
            }

            /** @brief Writes @p first, @p first + 1, ... into the parameters, in pack order. */
            void fill(float first)
            {
                float value = first;
                for (SushiBLAS::Tensor& parameter : parameters_)
                {
                    float* const elements = parameter.data_as<float>();
                    for (std::int64_t index = 0; index < parameter.num_elements; ++index)
                        elements[index] = value++;
                }
            }

            /** @brief Returns the parameters at @p indices, in that order. */
            std::vector<SushiBLAS::Tensor> parameters_at(
                const std::vector<std::size_t>& indices) const
            {
                std::vector<SushiBLAS::Tensor> chosen;
                for (const std::size_t index : indices)
                    chosen.push_back(parameters_[index]);
                return chosen;
            }

            /** @brief Returns the layout of the parameters. */
            SushiBLAS::PackLayout layout() const
            {
                return SushiBLAS::PackLayout::of(SushiRuntime::span<const SushiBLAS::Tensor>(
                    parameters_.data(), parameters_.size()));
            }

            /** @brief Expects the pack of @p publish to hold the parameters' current bytes. */
            void expect_pack_holds_parameters(const ParameterPublish& publish) const
            {
                const unsigned char* const base =
                    static_cast<const unsigned char*>(publish.pack().data());
                const std::vector<SushiBLAS::PackEntry>& entries =
                    publish.pack().layout().entries();
                ASSERT_EQ(entries.size(), parameters_.size());
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    EXPECT_EQ(std::memcmp(base + entries[index].offset_bytes,
                                          parameters_[index].data(), entries[index].size_bytes),
                              0)
                        << "parameter " << index;
                }
            }

            /// @brief The fabric the master listens on; it dies last.
            Distributed::InProcessFabric fabric_;

            /// @brief The master, attached to the runtime of the program from SetUp to TearDown.
            std::unique_ptr<Distributed::Cluster> cluster_;

            /// @brief The binding the pack is shared through.
            std::unique_ptr<SushiBLAS::ClusterBinding> binding_;

            /// @brief The parameters, in pack order.
            std::vector<SushiBLAS::Tensor> parameters_;
    };

    /** @brief Holds a parameter list the pack refuses and the words the refusal must name. */
    struct RefusalRow
    {
        /** @brief Names the row in the test report. */
        const char* name;

        /** @brief Holds the indices of the parameters handed to make(), in order. */
        std::vector<std::size_t> indices;

        /** @brief Holds the substring the refusal must name. */
        const char* needle;
    };

    /** @brief Lists the parameter lists that differ from the layout. */
    std::vector<RefusalRow> refusal_rows()
    {
        return {
            {"ReversedOrder", {1, 0}, "Parameter 0"},
            {"OneTensor", {0}, "2 tensors"},
        };
    }

    /** @brief Runs each parameter list make() refuses. */
    class Medium_ParameterPublishRefusal : public Medium_ParameterPublish,
                                           public ::testing::WithParamInterface<RefusalRow>
    {

    };
} // namespace

/** @brief Tests that make() shares the pack through the binding. */
TEST_F(Medium_ParameterPublish, MakeSharesThePack)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ParameterPublish> publish,
                            ParameterPublish::make(backend(), *binding_, layout(), parameters_));

    EXPECT_TRUE(binding_->is_shared(publish->pack()));
}

/** @brief Tests that make() copies the parameters into the pack. */
TEST_F(Medium_ParameterPublish, MakeCopiesTheParametersIntoThePack)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ParameterPublish> publish,
                            ParameterPublish::make(backend(), *binding_, layout(), parameters_));

    expect_pack_holds_parameters(*publish);
}

/** @brief Tests that each replay gathers the parameters as they are when it runs. */
TEST_F(Medium_ParameterPublish, AReplayGathersTheNewParameters)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ParameterPublish> publish,
                            ParameterPublish::make(backend(), *binding_, layout(), parameters_));
    const Graph::Graph empty;
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Graph::CompiledStep> step,
        Graph::CompiledStep::compile(backend(), empty, pool, publish.get()));
    EXPECT_EQ(step->operation_count(), 1u);

    fill(-4.0f);
    SA_EXPECT_OK(step->run());
    expect_pack_holds_parameters(*publish);
}

/** @brief Tests that make() refuses a parameter list that differs from the layout, naming it. */
TEST_P(Medium_ParameterPublishRefusal, MakeRefusesParametersThatDifferFromTheLayout)
{
    const RefusalRow& row = GetParam();
    const std::vector<SushiBLAS::Tensor> listed = parameters_at(row.indices);

    SA_EXPECT_FAILURE(ParameterPublish::make(backend(), *binding_, layout(), listed),
                      StatusCode::InvalidDescription, row.needle);
}

INSTANTIATE_TEST_SUITE_P(Lists, Medium_ParameterPublishRefusal, ::testing::ValuesIn(refusal_rows()),
                         [](const ::testing::TestParamInfo<RefusalRow>& info)
                         {
                             return ::SushiAI::Test::parameter_name(info.param.name);
                         });
