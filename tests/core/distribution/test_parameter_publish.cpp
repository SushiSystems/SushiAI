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
 * @file test_parameter_publish.cpp
 * @brief Tests ParameterPublish: the first copy, the gather of each replay and its refusals.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstring>
#include <memory>
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
#include <SushiRuntime/distributed/in_process_transport.hpp>
#include <SushiRuntime/distributed/kernel_registry.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/distribution/parameter_publish.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/shape/shape.hpp>

namespace
{
    using namespace SushiAI;
    using namespace SushiAI::Distribution;
    namespace Distributed = SushiRuntime::Distributed;

    /**
     * @brief Holds a master with no worker, its binding, and two float32 parameters of 3 and
     *        2 elements.
     */
    class Medium_ParameterPublish : public ::testing::Test
    {
        protected:
            /** @brief Attaches the master and allocates the parameters. */
            Medium_ParameterPublish()
                : runtime_(SushiRuntime::Runtime::create().value()), engine_(runtime_),
                  cluster_(Distributed::Cluster::attach(
                               runtime_, Distributed::InProcessTransport::make_master(fabric_))
                               .value()),
                  binding_(cluster_, SushiRuntime::make_sushi<Distributed::KernelRegistry>())
            {
                parameters_.push_back(allocate_tensor(engine_, Shape{3}, DType::Float32).value());
                parameters_.push_back(allocate_tensor(engine_, Shape{2}, DType::Float32).value());
                fill(1.0f);
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

            /// @brief The master's runtime.
            SushiRuntime::Runtime runtime_;

            /// @brief The engine over runtime_.
            SushiBLAS::Engine engine_;

            /// @brief The master.
            Distributed::Cluster cluster_;

            /// @brief The binding the pack is shared through.
            SushiBLAS::ClusterBinding binding_;

            /// @brief The parameters, in pack order.
            std::vector<SushiBLAS::Tensor> parameters_;
    };
} // namespace

/** @brief Tests that make() shares the pack and copies the parameters into it. */
TEST_F(Medium_ParameterPublish, MakeSharesThePackAndCopiesTheParameters)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ParameterPublish> publish,
                            ParameterPublish::make(engine_, binding_, layout(), parameters_));

    EXPECT_TRUE(binding_.is_shared(publish->pack()));
    expect_pack_holds_parameters(*publish);
}

/** @brief Tests that each replay gathers the parameters as they are when it runs. */
TEST_F(Medium_ParameterPublish, AReplayGathersTheNewParameters)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<ParameterPublish> publish,
                            ParameterPublish::make(engine_, binding_, layout(), parameters_));
    const Graph::Graph empty;
    Graph::TensorPool pool(engine_);
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Graph::CompiledStep> step,
        Graph::CompiledStep::compile(engine_, empty, pool, publish.get()));
    EXPECT_EQ(step->operation_count(), 1u);

    fill(-4.0f);
    SA_EXPECT_OK(step->run());
    expect_pack_holds_parameters(*publish);
}

/** @brief Tests that make() refuses a parameter list that differs from the layout. */
TEST_F(Medium_ParameterPublish, MakeRefusesParametersThatDifferFromTheLayout)
{
    const std::vector<SushiBLAS::Tensor> reversed = {parameters_[1], parameters_[0]};
    SA_EXPECT_FAILURE(ParameterPublish::make(engine_, binding_, layout(), reversed),
                      StatusCode::InvalidDescription, "Parameter 0");

    const std::vector<SushiBLAS::Tensor> one = {parameters_[0]};
    SA_EXPECT_FAILURE(ParameterPublish::make(engine_, binding_, layout(), one),
                      StatusCode::InvalidDescription, "2 tensors");
}
