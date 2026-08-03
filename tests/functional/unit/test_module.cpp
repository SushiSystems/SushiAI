/**************************************************************************/
/* test_module.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the layer library is host-lane, so the parameter walk - the
// thing that decides whether the optimizer sees a layer at all - is testable
// without hardware.

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>

using namespace SushiAI;
using NN::Linear;
using NN::Parameter;
using NN::ReLU;
using NN::Sequential;

namespace
{
    /** @brief Collects the walk's output as "name:extents" strings. */
    template<typename M>
    std::vector<std::string> walk(M& model)
    {
        std::vector<std::string> names;
        NN::for_each_parameter(model, [&names](const std::string& name, Parameter& parameter)
        {
            names.push_back(name + ":" + parameter.shape().to_string());
        });
        return names;
    }
}

TEST(Unit_Module, AnUntracedLayerYieldsNoParameters)
{
    // Parameters gain a shape and an id when forward() first runs, so the walk
    // over an untraced model is legitimately empty rather than full of blanks.
    Linear layer{4, 3};
    EXPECT_TRUE(walk(layer).empty());
    EXPECT_EQ(NN::parameter_count(layer), 0u);
}

TEST(Unit_Module, ALinearLayerNamesItsWeightAndBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 4]", "bias:[3]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 3 * 4 + 3);
}

TEST(Unit_Module, ABiasFreeLayerSimplyHasNoBiasEntry)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3, false};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 4]"}));
}

TEST(Unit_Module, SequentialNamesParametersByLayerIndex)
{
    // PyTorch's convention: the index is the layer's position, so a
    // parameterless activation leaves a gap and "2.weight" still names the
    // third layer. A checkpoint diff against a reference stays a diff.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Sequential model{Linear{4, 8}, ReLU{}, Linear{8, 3}};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)model.forward(builder, x);

    EXPECT_EQ(walk(model), (std::vector<std::string>{
        "0.weight:[8, 4]", "0.bias:[8]",
        "2.weight:[3, 8]", "2.bias:[3]",
    }));
}

TEST(Unit_Module, NestedContainersCompoundTheirIndices)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Sequential inner{Linear{4, 4}, ReLU{}};
    Sequential model{inner, Linear{4, 2}};

    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)model.forward(builder, x);

    EXPECT_EQ(walk(model), (std::vector<std::string>{
        "0.0.weight:[4, 4]", "0.0.bias:[4]",
        "1.weight:[2, 4]", "1.bias:[2]",
    }));
}

TEST(Unit_Module, TracingTwiceDoesNotDuplicateParameters)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");

    (void)layer.forward(builder, x);
    const Graph::ValueId first_weight = layer.weight.id();
    const std::size_t values_after_first = graph.value_count();

    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.id(), first_weight);
    // The second trace emits nodes but declares no new parameters, so the
    // value count grows by exactly the two intermediates.
    EXPECT_EQ(graph.value_count(), values_after_first + 2);
    EXPECT_EQ(NN::parameter_count(layer), 2u);
}

TEST(Unit_Module, ParametersAreDeclaredAsGradientRequiringLeaves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_TRUE(graph.value(layer.weight.id()).requires_grad);
    EXPECT_TRUE(graph.value(layer.bias.id()).requires_grad);
    EXPECT_EQ(graph.value(layer.weight.id()).lifetime, Graph::Lifetime::PARAMETER);
    EXPECT_FALSE(graph.value(x).requires_grad);
}

TEST(Unit_Module, RolesAndFansAreRecordedForTheInitializer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::WEIGHT);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::BIAS);
    EXPECT_EQ(layer.weight.fan_in(), 4);
    EXPECT_EQ(layer.weight.fan_out(), 3);
}

TEST(Unit_Module, LinearRejectsAnInputWhoseInnerExtentIsWrong)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueId x = builder.input(Shape{2, 7}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}
