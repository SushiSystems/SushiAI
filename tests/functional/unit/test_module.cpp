/****************************************************************/
/* test_module.cpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file test_module.cpp
 * @brief Tests the parameter walk over layers, containers and a layer enumerated at run time.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
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
    // An untraced layer has no parameters to walk; see tests/functional/unit/README.md.
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
    // The index is the layer's position, as in PyTorch; see tests/functional/unit/README.md.
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
    // The second trace declares no new parameter; see tests/functional/unit/README.md.
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

// Matches docs/getting_started/TUTORIAL.md §6 verbatim; see tests/functional/unit/README.md.

namespace
{
    /** @brief Holds docs/getting_started/TUTORIAL.md §6's learnable per-feature offset. */
    struct MyOffset
    {
        int64_t width = 0;
        NN::Parameter offset{};

        static constexpr auto fields()
        {
            return std::tuple{ NN::field("offset", &MyOffset::offset) };
        }

        Graph::ValueId forward(const Graph::GraphBuilder& builder, Graph::ValueId x)
        {
            const Graph::ValueId b = offset.declare(
                builder, Shape{width}, NN::ParameterRole::BIAS,
                width, width, "offset");
            return builder.bias_add(x, b);
        }
    };
}

TEST(Unit_Module, TheTutorialsCustomLayerTracesAndIsWalkable)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MyOffset layer{4};
    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    const Graph::ValueId y = layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"offset:[4]"}));
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));

    // The role decides the initialiser; see tests/functional/unit/README.md.
    EXPECT_EQ(layer.offset.role(), NN::ParameterRole::BIAS);

    // It satisfies the Module contract, so it composes.
    static_assert(NN::Module<MyOffset>, "the tutorial's layer must be a Module");
}

TEST(Unit_Module, TheTutorialIsRightThatMulWouldNotBroadcast)
{
    // mul goes through require_same and never broadcasts; see tests/functional/unit/README.md.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    const Graph::ValueId b = builder.input(Shape{4}, "b");

    EXPECT_THROW((void)builder.mul(x, b), Error);
    EXPECT_NO_THROW((void)builder.bias_add(x, b));
}

// WalkableStub is deliberately not a LayerStack; see tests/functional/unit/README.md.

namespace
{
    /** @brief A leaf that publishes an empty fields() and walks itself instead. */
    class WalkableStub
    {
        public:
            WalkableStub()
            {
                weight_.declare(builder_, Shape{2, 3}, NN::ParameterRole::WEIGHT, 3, 2, "w");
                bias_.declare(builder_, Shape{2}, NN::ParameterRole::BIAS, 0, 0, "b");
            }

            static constexpr auto fields() { return std::tuple<>{}; }

            Graph::ValueId forward(const Graph::GraphBuilder&, Graph::ValueId x) { return x; }

            void walk_parameters(std::string_view prefix, const NN::ParameterVisitor& visitor)
            {
                visitor(NN::Detail::join(prefix, "weight"), weight_);
                visitor(NN::Detail::join(prefix, "bias"), bias_);
            }

        private:
            Graph::Graph graph_{};
            Graph::GraphBuilder builder_{graph_};
            Parameter weight_{};
            Parameter bias_{};
    };

    /** @brief A container holding a walkable child, to check the two arms compose. */
    class StubHolder
    {
        public:
            static constexpr auto fields() { return std::tuple<>{}; }

            Graph::ValueId forward(const Graph::GraphBuilder&, Graph::ValueId x) { return x; }

            template<typename Fn>
            void visit_children(Fn&& visitor) { visitor(std::size_t{0}, child_); }

        private:
            WalkableStub child_{};
    };
}

TEST(Unit_Module, AWalkableLeafIsEnumeratedByItsOwnWalk)
{
    WalkableStub stub;
    EXPECT_EQ(walk(stub), (std::vector<std::string>{"weight:[2, 3]", "bias:[2]"}));
    EXPECT_EQ(NN::parameter_count(stub), 2u);
    EXPECT_EQ(NN::parameter_elements(stub), 8);
}

TEST(Unit_Module, AContainerPrefixesAWalkableChildTheSameWayItPrefixesAnyOther)
{
    // The container supplies "0.", the child the leaf names; see tests/functional/unit/README.md.
    StubHolder holder;
    EXPECT_EQ(walk(holder), (std::vector<std::string>{"0.weight:[2, 3]", "0.bias:[2]"}));
}

TEST(Unit_Module, TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt)
{
    // layers.hpp asserts the same at build time; see tests/functional/unit/README.md.
    static_assert(NN::Detail::HasParameterWalk<WalkableStub>::value,
                  "the stub must be detected as walking itself");
    static_assert(!NN::Detail::HasParameterWalk<Linear>::value,
                  "a compile-time leaf must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<Sequential<Linear, ReLU>>::value,
                  "a compile-time container must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<StubHolder>::value,
                  "a container holding a walkable child is still a container");
}
