/**************************************************************************/
/* test_layer.cpp                                                         */
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
 * @file test_layer.cpp
 * @brief Tests the run-time Layer interface and the LayerAdapter that wraps a Module in it.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::LayerAdapter;
using NN::Linear;

namespace
{
    /** @brief Implements Layer by hand with a fixed rank-3 shape and no parameters. */
    class FixedShapeLayer final : public NN::Layer
    {
        public:
            /** @brief Returns @p x unchanged. */
            Graph::ValueID forward(const Graph::GraphBuilder&, Graph::ValueID x) override
            {
                return x;
            }

            /** @brief Visits nothing; the layer owns no parameter. */
            void walk_parameters(std::string_view, const NN::ParameterVisitor&) override
            {

            }

            /** @brief Returns [2, 3, 4]. */
            Shape output_shape() const noexcept override
            {
                return Shape{2, 3, 4};
            }
    };

    /** @brief Collects the name of every parameter @p layer walks under @p prefix. */
    std::vector<std::string> parameter_names(NN::Layer& layer, std::string_view prefix)
    {
        std::vector<std::string> names;
        auto collect = [&names](const std::string& name, NN::Parameter&)
        {
            names.push_back(name);
        };
        layer.walk_parameters(prefix, NN::ParameterVisitor(collect));
        return names;
    }
} // namespace

/** @brief Tests that output_features() counts the elements of any implementation's shape. */
TEST(Small_Layer, OutputFeaturesCountsTheShapeOfAnyImplementation)
{
    const FixedShapeLayer layer;
    const NN::Layer& seen = layer;

    EXPECT_EQ(seen.output_features(), 2 * 3 * 4);
}

/** @brief Tests that an adapter traces through the interface what its module traces alone. */
TEST(Small_Layer, AnAdapterTracesWhatItsModuleTraces)
{
    Graph::Graph adapted_graph;
    const Graph::GraphBuilder adapted_builder(adapted_graph);
    LayerAdapter<Linear> adapter(Linear{6, 8}, 8);
    NN::Layer& layer = adapter;
    const Graph::ValueID adapted =
        layer.forward(adapted_builder, adapted_builder.input(Shape{2, 6}, "x"));

    Graph::Graph direct_graph;
    const Graph::GraphBuilder direct_builder(direct_graph);
    Linear module{6, 8};
    const Graph::ValueID direct =
        module.forward(direct_builder, direct_builder.input(Shape{2, 6}, "x"));

    EXPECT_EQ(adapted_graph.node_count(), direct_graph.node_count());
    EXPECT_EQ(adapted_graph.value(adapted).shape, direct_graph.value(direct).shape);
    EXPECT_EQ(adapted_graph.value(adapted).shape, (Shape{2, 8}));
}

/** @brief Tests that an adapter walks the parameters its module declared, under the prefix. */
TEST(Small_Layer, AnAdapterWalksItsModulesParametersUnderThePrefix)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    LayerAdapter<Linear> adapter(Linear{6, 8}, 8);
    NN::Layer& layer = adapter;
    static_cast<void>(layer.forward(builder, builder.input(Shape{2, 6}, "x")));

    EXPECT_EQ(parameter_names(layer, "block"),
              (std::vector<std::string>{"block.weight", "block.bias"}));
}

/** @brief Tests that the adapter hands back the module it wraps, fields intact. */
TEST(Small_Layer, TheAdapterHandsBackTheModuleItWraps)
{
    LayerAdapter<Linear> adapter(Linear{6, 8}, 8);
    const LayerAdapter<Linear>& view = adapter;

    EXPECT_EQ(adapter.layer().in_features, 6);
    EXPECT_EQ(view.layer().out_features, 8);
    EXPECT_EQ(&view.layer(), &adapter.layer());
}
