/****************************************************************/
/* test_layer_stack.cpp                                         */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Device-free: a LayerStack is host-lane, and the property this file exists to
// pin - that it names its parameters exactly as Sequential does - is a
// property of the walk, not of any memory.
//
// The comparison is written as "trace both, walk both, compare the two lists"
// rather than as "check LayerStack produces these literal names", so it fails
// if EITHER path changes its naming. A test against literals would let both
// drift together and still pass.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/ilayer.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using NN::GELU;
using NN::LayerAdapter;
using NN::LayerStack;
using NN::Linear;
using NN::Sequential;

namespace
{
    /** @brief Traces @p model and returns its parameters as "name:extents". */
    template<typename M>
    std::vector<std::string> traced_names(M& model, int64_t batch, int64_t features)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueId x = builder.input(Shape{batch, features}, "inputs");
        model.forward(builder, x);

        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, NN::Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }

    /** @brief 6 -> 8 -> gelu -> 4, as a runtime stack. */
    LayerStack make_stack()
    {
        LayerStack stack;
        stack.push(std::make_unique<LayerAdapter<Linear>>(Linear{6, 8}, 8));
        stack.push(std::make_unique<LayerAdapter<GELU>>(GELU{}, 8));
        stack.push(std::make_unique<LayerAdapter<Linear>>(Linear{8, 4}, 4));
        return stack;
    }
}

TEST(Unit_LayerStack, NamesItsParametersExactlyAsSequentialDoes)
{
    Sequential compile_time{Linear{6, 8}, GELU{}, Linear{8, 4}};
    LayerStack run_time = make_stack();

    const std::vector<std::string> expected = traced_names(compile_time, 4, 6);
    EXPECT_EQ(traced_names(run_time, 4, 6), expected);

    // Stated once as literals too, so the shared list above cannot become
    // trivially equal by both paths going empty.
    EXPECT_EQ(expected, (std::vector<std::string>{
        "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]", "2.bias:[4]"}));
}

TEST(Unit_LayerStack, KeepsTheIndexGapAParameterlessActivationLeaves)
{
    // The gap is the whole reason the naming matches: indices track layer
    // positions, not parameter positions. A stack that renumbered would
    // produce "1.weight" for the second Linear and load nothing.
    LayerStack stack = make_stack();
    const std::vector<std::string> names = traced_names(stack, 4, 6);

    ASSERT_EQ(names.size(), 4u);
    EXPECT_EQ(names[2].substr(0, 2), "2.");
}

TEST(Unit_LayerStack, ReportsTheLastLayersOutputWidth)
{
    LayerStack stack = make_stack();
    EXPECT_EQ(stack.size(), 3u);
    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(stack.output_features(), 4);
}

TEST(Unit_LayerStack, TracesTheLayersInOrder)
{
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");
    const Graph::ValueId y = stack.forward(builder, x);

    EXPECT_EQ(graph.value(y).shape, (Shape{4, 4}));
}

TEST(Unit_LayerStack, RefusesToTraceWhenEmpty)
{
    LayerStack stack;
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");

    SA_EXPECT_ERROR((void)stack.forward(builder, x), "no layers", "identity");
}

TEST(Unit_LayerStack, RefusesAnOutputWidthWhenEmpty)
{
    const LayerStack stack;
    SA_EXPECT_ERROR((void)stack.output_features(), "no layers");
}

TEST(Unit_LayerStack, RefusesANullLayer)
{
    LayerStack stack;
    SA_EXPECT_ERROR(stack.push(nullptr), "null layer", "index 0");
}

TEST(Unit_LayerStack, ReportsAnOutOfRangeLayerIndex)
{
    const LayerStack stack = make_stack();
    EXPECT_NO_THROW((void)stack.layer(2));
    SA_EXPECT_ERROR((void)stack.layer(3), "layer 3", "holds 3");
}

TEST(Unit_LayerStack, TracingTwiceDoesNotDuplicateParameters)
{
    // Parameter::declare is idempotent, and the runtime path must inherit that
    // rather than reintroducing the duplicate-on-retrace bug at this layer.
    // Both traces go into ONE graph, because a parameter's id names a value in
    // the graph that declared it - retracing into a second graph would be
    // reusing an index that means something else there.
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);
}
