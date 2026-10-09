/**************************************************************************/
/* test_init_state.cpp                                                    */
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
 * @file test_init_state.cpp
 * @brief Tests the initial fill that NN::initialize() records over a ModelState.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <gtest/gtest.h>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include "../../common/test_status.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;

namespace
{
    class Integration_InitState : public SaTest::EngineTest {};

    /** @brief The absolute tolerance of a value that is stored and read back unchanged. */
    constexpr double K_STORED_TOLERANCE = 0.0;

    /** @brief The architecture every case that needs a parameter model traces. */
    using Mlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    Mlp make_mlp()
    {
        return Mlp(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    /** @brief Traces @p model into @p graph from a [4, 6] input. */
    void trace_mlp(Mlp& model, Graph::Graph& graph)
    {
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
        static_cast<void>(model.forward(builder, x));
    }

    /** @brief Adds a BatchNorm2d named "bn" over @p channels and marks it as the output. */
    void build_batch_norm(NN::ModelGraph& model, int64_t channels)
    {
        SA_ASSERT_OK_AND_ASSIGN(const auto norm,
                                model.add("bn", NN::BatchNorm2d(channels, NN::NormOptions{}),
                                          model.input()));
        SA_EXPECT_OK(model.output("normalised", norm));
    }

    /** @brief Adds a 3-to-4 channel convolution named "conv" followed by a BatchNorm2d "bn". */
    void build_conv_norm(NN::ModelGraph& model)
    {
        SA_ASSERT_OK_AND_ASSIGN(const auto conv,
                                model.add("conv", NN::Conv2d::same_padded(3, 4, 3, 1),
                                          model.input()));
        SA_ASSERT_OK_AND_ASSIGN(const auto norm,
                                model.add("bn", NN::BatchNorm2d(4, NN::NormOptions{}), conv));
        SA_EXPECT_OK(model.output("normalised", norm));
    }

    /** @brief Traces @p model in training mode from a batch of @p input_shape. */
    void trace_training(NN::ModelGraph& model, const Shape& input_shape)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(input_shape, "x");
        NN::BufferUpdateList list;
        SA_EXPECT_OK(model.trace(NN::Trace::training(builder, list), x));
    }

    /** @brief Expects every element of the tensor named @p name to be @p expected. */
    void expect_every_element_near(NN::ModelState& state, std::string_view name, double expected)
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(name));
        for (const double value : SaTest::read_tensor(*tensor))
            EXPECT_NEAR(value, expected, K_STORED_TOLERANCE);
    }
} // namespace

/**
 * @brief Tests that a scale starts at one, a shift at zero, and a buffer at its initial value.
 */
TEST_F(Integration_InitState, AScaleIsOneAShiftIsZeroAndABufferHoldsItsInitialValue)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    build_batch_norm(model, 3);
    trace_training(model, Shape{2, 4, 4, 3});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    expect_every_element_near(state, "bn.weight", 1.0);
    expect_every_element_near(state, "bn.bias", 0.0);
    expect_every_element_near(state, "bn.running_mean", 0.0);
    expect_every_element_near(state, "bn.running_var", 1.0);
}

/**
 * @brief Tests that zeroing the final layer zeroes the last fan-scaled entry and nothing after.
 */
TEST_F(Integration_InitState, ZeroFinalLayerZeroesOnlyTheLastFanScaledEntry)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    build_conv_norm(model);
    trace_training(model, Shape{2, 4, 4, 3});

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    expect_every_element_near(state, "conv.weight", 0.0);
    expect_every_element_near(state, "bn.weight", 1.0);
}

/**
 * @brief Tests that without the zeroed final layer, the last weight holds a nonzero element.
 */
TEST_F(Integration_InitState, WithoutZeroFinalLayerTheLastWeightIsNotZero)
{
    Mlp model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    SA_EXPECT_OK(NN::initialize(*engine, state, policy));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* last, state.tensor("2.weight"));
    bool any_nonzero = false;
    for (const double value : SaTest::read_tensor(*last))
        if (std::abs(value) > K_STORED_TOLERANCE) any_nonzero = true;
    EXPECT_TRUE(any_nonzero);
}

/**
 * @brief Tests that two states of equal models, filled with one seed, hold equal bits.
 */
TEST_F(Integration_InitState, TheSameSeedGivesTheSameBits)
{
    Mlp first_model = make_mlp();
    Graph::Graph first_graph;
    trace_mlp(first_model, first_graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState first, NN::ModelState::create(*engine, first_model));

    Mlp second_model = make_mlp();
    Graph::Graph second_graph;
    trace_mlp(second_model, second_graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState second,
                            NN::ModelState::create(*engine, second_model));

    SA_EXPECT_OK(NN::initialize(*engine, first));
    SA_EXPECT_OK(NN::initialize(*engine, second));

    for (const NN::StateEntry& entry : first.entries())
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* a, first.tensor(entry.name));
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* b, second.tensor(entry.name));
        EXPECT_EQ(SaTest::read_tensor_bytes(*a), SaTest::read_tensor_bytes(*b));
    }
}
