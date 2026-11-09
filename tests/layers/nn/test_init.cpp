/**************************************************************************/
/* test_init.cpp                                                          */
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
 * @file test_init.cpp
 * @brief Tests the initial fill that NN::initialize() records over a ModelState.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_InitState : public SaTest::EngineTest {};

    /** @brief The architecture every case that needs a parameter model traces. */
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    MLP make_mlp()
    {
        return MLP(NN::Linear{6, 8}, NN::GELU{}, NN::Linear{8, 4});
    }

    /** @brief Traces @p model into @p graph from a [4, 6] input. */
    void trace_mlp(MLP& model, Graph::Graph& graph)
    {
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{4, 6}, "inputs");
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
        const Graph::ValueID x = builder.input(input_shape, "x");
        NN::BufferUpdateList list;
        SA_EXPECT_OK(model.trace(NN::Trace::training(builder, list), x));
    }

    /** @brief Expects every element of the tensor named @p name to equal @p expected. */
    void expect_every_element_equal(NN::ModelState& state, std::string_view name, double expected)
    {
        SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* tensor, state.tensor(name));
        const std::vector<double> values = SaTest::read_tensor(*tensor);
        EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                                [expected](double value) { return value == expected; }))
            << "tensor " << name;
    }

    /** @brief Returns the mean of @p values. */
    double sample_mean(const std::vector<double>& values)
    {
        return std::accumulate(values.begin(), values.end(), 0.0) /
               static_cast<double>(values.size());
    }

    /** @brief Returns the deviation of @p values around @p mean, over all of them. */
    double sample_deviation(const std::vector<double>& values, double mean)
    {
        const double squares = std::accumulate(values.begin(), values.end(), 0.0,
                                               [mean](double sum, double value)
                                               {
                                                   return sum + (value - mean) * (value - mean);
                                               });

        return std::sqrt(squares / static_cast<double>(values.size()));
    }

    /**
     * @brief Holds the number of standard errors a statistic may stray: the two-sided normal tail
     * beyond 6.1 is 1.0e-9, so a correct fill of a fixed seed fails a check with that chance.
     */
    constexpr double K_STANDARD_ERRORS = 6.1;

    /**
     * @brief Bounds the distance of the mean of @p count normal draws of deviation @p sigma from
     * zero: the mean has the standard error sigma / sqrt(count).
     */
    double mean_bound(std::size_t count, double sigma)
    {
        return K_STANDARD_ERRORS * sigma / std::sqrt(static_cast<double>(count));
    }

    /**
     * @brief Bounds the distance of the deviation of @p count normal draws from @p sigma: the
     * sample deviation has the standard error sigma / sqrt(2 * count) for a large count.
     */
    double deviation_bound(std::size_t count, double sigma)
    {
        return K_STANDARD_ERRORS * sigma / std::sqrt(2.0 * static_cast<double>(count));
    }

    /** @brief Builds a three-channel BatchNorm2d model and traces it, ready for a state. */
    void prepare_batch_norm(NN::ModelGraph& model)
    {
        build_batch_norm(model, 3);
        trace_training(model, Shape{2, 4, 4, 3});
    }
} // namespace

/** @brief Tests that a scale starts at one. */
TEST_F(Medium_InitState, AScaleStartsAtOne)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    prepare_batch_norm(model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    SA_EXPECT_OK(NN::initialize(backend(), state));

    expect_every_element_equal(state, "bn.weight", 1.0);
}

/** @brief Tests that a shift starts at zero. */
TEST_F(Medium_InitState, AShiftStartsAtZero)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    prepare_batch_norm(model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    SA_EXPECT_OK(NN::initialize(backend(), state));

    expect_every_element_equal(state, "bn.bias", 0.0);
}

/** @brief Tests that a buffer holds its initial value: the running mean zero, the variance one. */
TEST_F(Medium_InitState, ABufferHoldsItsInitialValue)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    prepare_batch_norm(model);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    SA_EXPECT_OK(NN::initialize(backend(), state));

    expect_every_element_equal(state, "bn.running_mean", 0.0);
    expect_every_element_equal(state, "bn.running_var", 1.0);
}

/**
 * @brief Tests that zeroing the final layer zeroes every element of the convolution weight and
 *        leaves the normalisation scale at one.
 */
TEST_F(Medium_InitState, ZeroFinalLayerZeroesTheConvWeightAndKeepsTheNormScaleAtOne)
{
    NN::ModelGraph model(Shape{4, 4, 3});
    build_conv_norm(model);
    trace_training(model, Shape{2, 4, 4, 3});
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    SA_EXPECT_OK(NN::initialize(backend(), state));

    expect_every_element_equal(state, "conv.weight", 0.0);
    expect_every_element_equal(state, "bn.weight", 1.0);
}

/**
 * @brief Tests that without the zeroed final layer, the last weight holds a nonzero element.
 */
TEST_F(Medium_InitState, WithoutZeroFinalLayerTheLastWeightIsNotZero)
{
    MLP model = make_mlp();
    Graph::Graph graph;
    trace_mlp(model, graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    NN::InitPolicy policy;
    policy.zero_final_layer = false;

    SA_EXPECT_OK(NN::initialize(backend(), state, policy));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* last, state.tensor("2.weight"));
    const std::vector<double> values = SaTest::read_tensor(*last);
    EXPECT_TRUE(std::any_of(values.begin(), values.end(),
                            [](double value) { return value != 0.0; }));
}

namespace
{
/** @brief Holds the name of one state entry of the two-layer model. */
struct EntryRow
{
    std::string name;  /**< The row's name. */
    std::string entry; /**< The state entry compared. */
};
} // namespace

/** @brief Provides the Engine and the entry rows of the seed case. */
class Medium_InitStateSeed : public SaTest::EngineTest,
                             public ::testing::WithParamInterface<EntryRow>
{
};

/**
 * @brief Tests that two states of equal models, filled with one seed, hold equal bits in an entry.
 */
TEST_P(Medium_InitStateSeed, TheSameSeedGivesTheSameBits)
{
    MLP first_model = make_mlp();
    Graph::Graph first_graph;
    trace_mlp(first_model, first_graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState first, NN::ModelState::create(*engine, first_model));
    MLP second_model = make_mlp();
    Graph::Graph second_graph;
    trace_mlp(second_model, second_graph);
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState second,
                            NN::ModelState::create(*engine, second_model));
    SA_EXPECT_OK(NN::initialize(backend(), first));
    SA_EXPECT_OK(NN::initialize(backend(), second));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* a, first.tensor(GetParam().entry));
    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* b, second.tensor(GetParam().entry));

    EXPECT_EQ(SaTest::read_tensor_bytes(*a), SaTest::read_tensor_bytes(*b));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_InitStateSeed,
    ::testing::Values(EntryRow{"TheFirstWeight", "0.weight"}, EntryRow{"TheFirstBias", "0.bias"},
                      EntryRow{"TheLastWeight", "2.weight"}, EntryRow{"TheLastBias", "2.bias"}),
    [](const ::testing::TestParamInfo<EntryRow>& info) { return info.param.name; });

/**
 * @brief Tests that a normal fill of deviation 0.02 gives a table of that spread, zeroing or not.
 */
TEST_F(Medium_InitState, ANormalFillHasTheDeviationItStates)
{
    constexpr int64_t K_ROWS = 64;
    constexpr int64_t K_WIDTH = 32;
    NN::Embedding model{};
    model.num_embeddings = K_ROWS;
    model.embedding_dim = K_WIDTH;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID ids = builder.input(Shape{2, 4}, "ids", DType::Int32);
    static_cast<void>(model.forward(NN::Trace::evaluation(builder), ids));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));

    // The seed is the fixed default of InitPolicy, so the draw is the same on every run.
    SA_EXPECT_OK(NN::initialize(backend(), state));

    SA_ASSERT_OK_AND_ASSIGN(SushiBLAS::Tensor* table, state.tensor("weight"));
    const std::vector<double> values = SaTest::read_tensor(*table);
    ASSERT_EQ(values.size(), static_cast<std::size_t>(K_ROWS * K_WIDTH));
    const double mean = sample_mean(values);
    EXPECT_NEAR(mean, 0.0, mean_bound(values.size(), NN::K_EMBEDDING_DEVIATION));
    EXPECT_NEAR(sample_deviation(values, mean), NN::K_EMBEDDING_DEVIATION,
                deviation_bound(values.size(), NN::K_EMBEDDING_DEVIATION));
}

/** @brief Tests that a normal fill of a deviation that is not positive is refused. */
TEST_F(Medium_InitState, ANormalFillOfNoDeviationIsRefused)
{
    NN::Embedding model{};
    model.num_embeddings = 4;
    model.embedding_dim = 2;
    model.weight_init = NN::ParameterInit::normal(0.0);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID ids = builder.input(Shape{2}, "ids", DType::Int32);
    static_cast<void>(model.forward(NN::Trace::evaluation(builder), ids));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    SA_EXPECT_FAILURE(NN::initialize(backend(), state), StatusCode::InvalidDescription,
                      "positive, finite deviation");
}
