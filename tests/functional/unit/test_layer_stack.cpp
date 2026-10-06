/**************************************************************************/
/* test_layer_stack.cpp                                                   */
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
 * @file test_layer_stack.cpp
 * @brief Tests that a LayerStack names and declares its parameters exactly as Sequential does.
 * @author Mustafa Garip
 */

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

    // Literals keep both paths from going empty together; see tests/functional/unit/README.md.
    EXPECT_EQ(expected, (std::vector<std::string>{
        "0.weight:[8, 6]", "0.bias:[8]", "2.weight:[4, 8]", "2.bias:[4]"}));
}

TEST(Unit_LayerStack, KeepsTheIndexGapAParameterlessActivationLeaves)
{
    // Indices track layer positions, not parameter positions; see tests/functional/unit/README.md.
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
    // Both traces go into one graph; see tests/functional/unit/README.md.
    LayerStack stack = make_stack();

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueId x = builder.input(Shape{4, 6}, "inputs");

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);

    (void)stack.forward(builder, x);
    EXPECT_EQ(NN::parameter_count(stack), 4u);
}
