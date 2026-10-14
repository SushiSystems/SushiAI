/**************************************************************************/
/* test_module.cpp                                                        */
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
 * @file test_module.cpp
 * @brief Tests Parameter across graphs and the state walk over layers, containers and fields.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using namespace SushiAI;
using NN::Linear;
using NN::Parameter;
using NN::ReLU;
using NN::Sequential;

namespace
{
    /** @brief Holds the absolute tolerance of a double stored and read back unchanged. */
    constexpr double K_EXACT_TOLERANCE = 1e-15;

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

    /** @brief Traces @p layer over a batch of two and returns the result's shape. */
    template<typename L>
    Shape traced_shape(L& layer, int64_t features)
    {
        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);
        const Graph::ValueID x = builder.input(Shape{2, features}, "x");
        return graph.value(layer.forward(builder, x)).shape;
    }
}

TEST(Unit_Module, AnUntracedLayerYieldsNoParameters)
{
    // An untraced layer has no parameters to walk; see tests/unit/README.md.
    Linear layer{4, 3};
    EXPECT_TRUE(walk(layer).empty());
    EXPECT_EQ(NN::parameter_count(layer), 0u);
}

TEST(Unit_Module, ALinearLayerNamesItsWeightAndBias)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 4]", "bias:[3]"}));
    EXPECT_EQ(NN::parameter_elements(layer), 3 * 4 + 3);
}

TEST(Unit_Module, ABiasFreeLayerSimplyHasNoBiasEntry)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3, false};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"weight:[3, 4]"}));
}

TEST(Unit_Module, SequentialNamesParametersByLayerIndex)
{
    // The index is the layer's position, as in PyTorch; see tests/unit/README.md.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Sequential model{Linear{4, 8}, ReLU{}, Linear{8, 3}};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
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

    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
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
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    (void)layer.forward(builder, x);
    const Graph::ValueID first_weight = layer.weight.id_in(graph);
    const std::size_t values_after_first = graph.value_count();

    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.id_in(graph), first_weight);
    // The second trace declares no new parameter; see tests/unit/README.md.
    EXPECT_EQ(graph.value_count(), values_after_first + 2);
    EXPECT_EQ(NN::parameter_count(layer), 2u);
}

TEST(Unit_Module, ParametersAreDeclaredAsGradientRequiringLeaves)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_TRUE(graph.value(layer.weight.id_in(graph)).requires_grad);
    EXPECT_TRUE(graph.value(layer.bias.id_in(graph)).requires_grad);
    EXPECT_EQ(graph.value(layer.weight.id_in(graph)).lifetime, Graph::Lifetime::Parameter);
    EXPECT_FALSE(graph.value(x).requires_grad);
}

TEST(Unit_Module, RolesAndFansAreRecordedForTheInitializer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    (void)layer.forward(builder, x);

    EXPECT_EQ(layer.weight.role(), NN::ParameterRole::Weight);
    EXPECT_EQ(layer.bias.role(), NN::ParameterRole::Bias);
    EXPECT_EQ(layer.weight.fan_in(), 4);
    EXPECT_EQ(layer.weight.fan_out(), 3);
}

TEST(Unit_Module, LinearRejectsAnInputWhoseInnerExtentIsWrong)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Linear layer{4, 3};
    const Graph::ValueID x = builder.input(Shape{2, 7}, "x");

    EXPECT_THROW((void)layer.forward(builder, x), Error);
}

// Matches docs/getting_started/TUTORIAL.md §6 verbatim; see tests/unit/README.md.

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

        Graph::ValueID forward(const Graph::GraphBuilder& builder, Graph::ValueID x)
        {
            const Graph::ValueID b = offset.declare(
                builder, Shape{width}, NN::ParameterRole::Bias,
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
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const Graph::ValueID y = layer.forward(builder, x);

    EXPECT_EQ(walk(layer), (std::vector<std::string>{"offset:[4]"}));
    EXPECT_EQ(graph.value(y).shape, (Shape{2, 4}));

    // The role decides the initialiser; see tests/unit/README.md.
    EXPECT_EQ(layer.offset.role(), NN::ParameterRole::Bias);

    // It satisfies the Module contract, so it composes.
    static_assert(NN::Module<MyOffset>, "the tutorial's layer must be a Module");
}

TEST(Unit_Module, TheTutorialIsRightThatMulWouldNotBroadcast)
{
    // mul goes through require_same and never broadcasts; see tests/unit/README.md.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    const Graph::ValueID b = builder.input(Shape{4}, "b");

    EXPECT_THROW((void)builder.mul(x, b), Error);
    EXPECT_NO_THROW((void)builder.bias_add(x, b));
}

// WalkableStub is deliberately not a LayerStack; see tests/unit/README.md.

namespace
{
    /** @brief A leaf that publishes an empty fields() and walks itself instead. */
    class WalkableStub
    {
        public:
            WalkableStub()
            {
                weight_.declare(builder_, Shape{2, 3}, NN::ParameterRole::Weight, 3, 2, "w");
                bias_.declare(builder_, Shape{2}, NN::ParameterRole::Bias, 0, 0, "b");
            }

            static constexpr auto fields() { return std::tuple<>{}; }

            Graph::ValueID forward(const Graph::GraphBuilder&, Graph::ValueID x) { return x; }

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

            Graph::ValueID forward(const Graph::GraphBuilder&, Graph::ValueID x) { return x; }

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
    // The container supplies "0.", the child the leaf names; see tests/unit/README.md.
    StubHolder holder;
    EXPECT_EQ(walk(holder), (std::vector<std::string>{"0.weight:[2, 3]", "0.bias:[2]"}));
}

TEST(Unit_Module, TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt)
{
    // layers.hpp asserts the same at build time; see tests/unit/README.md.
    static_assert(NN::Detail::HasParameterWalk<WalkableStub>::value,
                  "the stub must be detected as walking itself");
    static_assert(!NN::Detail::HasParameterWalk<Linear>::value,
                  "a compile-time leaf must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<Sequential<Linear, ReLU>>::value,
                  "a compile-time container must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<StubHolder>::value,
                  "a container holding a walkable child is still a container");
}

/** @brief Tests that Linear maps the per-sample shape [in_features] to [out_features]. */
TEST(Unit_Module, LinearReportsItsOutFeaturesAsThePerSampleOutputShape)
{
    const Linear layer{4, 3};
    EXPECT_EQ(layer.output_shape(Shape{4}), (Shape{3}));
}

/** @brief Tests that Linear's output_shape() matches what forward() traces, batch axis aside. */
TEST(Unit_Module, LinearsOutputShapeAgreesWithItsTrace)
{
    Linear layer{4, 3};
    EXPECT_EQ(traced_shape(layer, 4), (Shape{2, layer.output_shape(Shape{4})[0]}));
}

/** @brief Tests that Linear refuses a per-sample shape whose rank is not 1. */
TEST(Unit_Module, LinearRefusesAPerSampleShapeOfTheWrongRank)
{
    const Linear layer{4, 3};

    // The first extent equals in_features, so only the rank check can refuse this shape.
    SA_EXPECT_ERROR((void)layer.output_shape(Shape{4, 2, 1}), "Linear(4, 3)", "[4, 2, 1]", "[4]");
}

/** @brief Tests that Linear refuses a rank-1 per-sample shape whose width is not in_features. */
TEST(Unit_Module, LinearRefusesAPerSampleShapeOfTheWrongWidth)
{
    const Linear layer{4, 3};

    SA_EXPECT_ERROR((void)layer.output_shape(Shape{7}), "Linear(4, 3)", "[7]", "[4]");
}

/** @brief Tests that every activation returns the per-sample shape it was given, at any rank. */
TEST(Unit_Module, AnActivationReportsItsInputShapeAsItsOutputShape)
{
    for (const Shape& input : {Shape{5}, Shape{4, 4, 3}})
    {
        EXPECT_EQ(ReLU{}.output_shape(input), input);
        EXPECT_EQ(NN::GELU{}.output_shape(input), input);
        EXPECT_EQ(NN::Tanh{}.output_shape(input), input);
        EXPECT_EQ(NN::Sigmoid{}.output_shape(input), input);
    }
}

/** @brief Tests that every activation's output_shape() matches what forward() traces. */
TEST(Unit_Module, AnActivationsOutputShapeAgreesWithItsTrace)
{
    ReLU relu;
    NN::GELU gelu;
    NN::Tanh tanh;
    NN::Sigmoid sigmoid;

    EXPECT_EQ(traced_shape(relu, 5), (Shape{2, relu.output_shape(Shape{5})[0]}));
    EXPECT_EQ(traced_shape(gelu, 5), (Shape{2, gelu.output_shape(Shape{5})[0]}));
    EXPECT_EQ(traced_shape(tanh, 5), (Shape{2, tanh.output_shape(Shape{5})[0]}));
    EXPECT_EQ(traced_shape(sigmoid, 5), (Shape{2, sigmoid.output_shape(Shape{5})[0]}));
}

namespace
{
    /** @brief Declares one [2, 3] weight into @p builder and returns its leaf. */
    Graph::ValueID declare_weight(Parameter& parameter, const Graph::GraphBuilder& builder)
    {
        return parameter.declare(builder, Shape{2, 3}, NN::ParameterRole::Weight, 3, 2, "weight");
    }
}

/** @brief Tests that one parameter traced into two graphs has its own leaf in each. */
TEST(Unit_Module, AParameterTracedIntoTwoGraphsHasALeafInEach)
{
    Graph::Graph first;
    Graph::Graph second;
    const Graph::GraphBuilder first_builder(first);
    const Graph::GraphBuilder second_builder(second);

    // The padding input gives the first graph's leaf an id the second graph does not repeat.
    (void)first_builder.input(Shape{2, 3}, "padding");

    Parameter parameter;
    const Graph::ValueID first_leaf = declare_weight(parameter, first_builder);
    const Graph::ValueID second_leaf = declare_weight(parameter, second_builder);

    EXPECT_EQ(first_leaf, 1u);
    EXPECT_EQ(second_leaf, 0u);
    EXPECT_EQ(parameter.id_in(first), first_leaf);
    EXPECT_EQ(parameter.id_in(second), second_leaf);
    EXPECT_EQ(first.state_leaf(parameter.state()), first_leaf);
    EXPECT_EQ(second.state_leaf(parameter.state()), second_leaf);
    EXPECT_EQ(second.value(second_leaf).lifetime, Graph::Lifetime::Parameter);
}

/** @brief Tests that id_in() reports NO_VALUE for a graph the parameter was never traced into. */
TEST(Unit_Module, AParameterHasNoLeafInAGraphItWasNotTracedInto)
{
    Graph::Graph traced;
    Graph::Graph untouched;
    const Graph::GraphBuilder builder(traced);

    Parameter parameter;
    EXPECT_EQ(parameter.id_in(traced), Graph::K_NO_VALUE);
    EXPECT_EQ(parameter.state(), Graph::K_NO_STATE);

    (void)declare_weight(parameter, builder);

    EXPECT_NE(parameter.state(), Graph::K_NO_STATE);
    EXPECT_EQ(parameter.id_in(untouched), Graph::K_NO_VALUE);
}

/** @brief Tests that a second declare in one graph returns the first leaf and adds no value. */
TEST(Unit_Module, ASecondDeclareInOneGraphReturnsTheFirstLeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Parameter parameter;
    const Graph::ValueID first_leaf = declare_weight(parameter, builder);
    const std::size_t values_after_first = graph.value_count();

    EXPECT_EQ(declare_weight(parameter, builder), first_leaf);
    EXPECT_EQ(graph.value_count(), values_after_first);
    EXPECT_EQ(graph.state_leaves().size(), 1u);
}

/** @brief Tests that a declare with another shape is refused, in the same graph and in another. */
TEST(Unit_Module, ADeclareWithAnotherShapeIsRefused)
{
    Graph::Graph graph;
    Graph::Graph other;
    const Graph::GraphBuilder builder(graph);
    const Graph::GraphBuilder other_builder(other);

    Parameter parameter;
    (void)declare_weight(parameter, builder);

    SA_EXPECT_ERROR((void)parameter.declare(builder, Shape{4, 3}, NN::ParameterRole::Weight,
                                            3, 4, "weight"),
                    "'weight'", "[2, 3]", "[4, 3]");
    SA_EXPECT_ERROR((void)parameter.declare(other_builder, Shape{4, 3}, NN::ParameterRole::Weight,
                                            3, 4, "weight"),
                    "'weight'", "[2, 3]", "[4, 3]");
    EXPECT_EQ(other.value_count(), 0u);
    EXPECT_EQ(parameter.shape(), (Shape{2, 3}));
}

/** @brief Tests that a declare with another element type is refused in either graph. */
TEST(Unit_Module, ADeclareWithAnotherElementTypeIsRefused)
{
    Graph::Graph graph;
    Graph::Graph other;
    const Graph::GraphBuilder builder(graph);
    const Graph::GraphBuilder other_builder(other);

    Parameter parameter;
    (void)declare_weight(parameter, builder);

    SA_EXPECT_ERROR((void)parameter.declare(builder, Shape{2, 3}, NN::ParameterRole::Weight,
                                            3, 2, "weight", DType::Float64),
                    "'weight'", dtype_name(DType::Float32), dtype_name(DType::Float64));
    SA_EXPECT_ERROR((void)parameter.declare(other_builder, Shape{2, 3},
                                            NN::ParameterRole::Weight, 3, 2, "weight",
                                            DType::Float64),
                    "'weight'", dtype_name(DType::Float32), dtype_name(DType::Float64));
    EXPECT_EQ(other.value_count(), 0u);
    EXPECT_EQ(parameter.dtype(), DType::Float32);
}

/** @brief Tests that a later declare with another fill and other fans leaves the first ones. */
TEST(Unit_Module, ALaterDeclareKeepsTheFirstDescription)
{
    Graph::Graph first;
    Graph::Graph second;
    const Graph::GraphBuilder first_builder(first);
    const Graph::GraphBuilder second_builder(second);

    Parameter scale;
    (void)scale.declare(first_builder, Shape{4}, NN::ParameterRole::Scale,
                        NN::ParameterInit::constant_value(1.5), 4, 4, "scale");
    (void)scale.declare(second_builder, Shape{4}, NN::ParameterRole::Weight,
                        NN::ParameterInit::fan_scaled(), 9, 9, "scale");

    EXPECT_EQ(scale.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(scale.init().constant, 1.5, K_EXACT_TOLERANCE);
    EXPECT_EQ(scale.role(), NN::ParameterRole::Scale);
    EXPECT_EQ(scale.fan_in(), 4);
    EXPECT_EQ(scale.fan_out(), 4);
    EXPECT_NE(scale.id_in(second), Graph::K_NO_VALUE);
}

/** @brief Tests that a copied layer is new state: its own StateID and its own leaf. */
TEST(Unit_Module, ACopiedLayerHasANewStateID)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    Linear original{4, 3};
    (void)original.forward(builder, x);

    Linear copy = original;
    EXPECT_EQ(copy.weight.state(), Graph::K_NO_STATE);
    EXPECT_EQ(copy.weight.id_in(graph), Graph::K_NO_VALUE);

    EXPECT_TRUE(copy.weight.declared());

    (void)copy.forward(builder, x);

    EXPECT_NE(copy.weight.id_in(graph), Graph::K_NO_VALUE);

    EXPECT_NE(copy.weight.state(), Graph::K_NO_STATE);
    EXPECT_NE(copy.weight.state(), original.weight.state());
    EXPECT_NE(copy.weight.id_in(graph), original.weight.id_in(graph));
    EXPECT_NE(copy.bias.id_in(graph), original.bias.id_in(graph));
}

/** @brief Tests that a moved layer keeps its StateID, so it still matches the graph it traced. */
TEST(Unit_Module, AMovedLayerKeepsItsStateID)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    Linear original{4, 3};
    (void)original.forward(builder, x);
    const Graph::StateID weight_state = original.weight.state();
    const Graph::ValueID weight_leaf = original.weight.id_in(graph);
    const std::size_t values_after_trace = graph.value_count();

    Linear moved = std::move(original);

    EXPECT_EQ(moved.weight.state(), weight_state);
    EXPECT_EQ(moved.weight.id_in(graph), weight_leaf);

    (void)moved.forward(builder, x);

    // A second trace of the moved layer adds the two activations and no parameter leaf.
    EXPECT_EQ(graph.value_count(), values_after_trace + 2);
}

/** @brief Tests that a role declare records the fill the role stands for. */
TEST(Unit_Module, ARoleDeclareRecordsTheFillOfItsRole)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Parameter weight;
    Parameter bias;
    (void)weight.declare(builder, Shape{2, 3}, NN::ParameterRole::Weight, 3, 2, "weight");
    (void)bias.declare(builder, Shape{2}, NN::ParameterRole::Bias, 3, 2, "bias");

    EXPECT_EQ(weight.init().kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_EQ(bias.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(bias.init().constant, 0.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that an init declare keeps the stated fill and the stated role, each as given. */
TEST(Unit_Module, AnInitDeclareRecordsTheStatedFillAndTheStatedRole)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Parameter offset;
    Parameter kernel;
    (void)offset.declare(builder, Shape{4}, NN::ParameterRole::Bias,
                         NN::ParameterInit::constant_value(1.5), 4, 4, "offset");
    (void)kernel.declare(builder, Shape{2, 3}, NN::ParameterRole::Weight,
                         NN::ParameterInit::fan_scaled(), 3, 2, "kernel");

    EXPECT_EQ(offset.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(offset.init().constant, 1.5, K_EXACT_TOLERANCE);
    EXPECT_EQ(offset.role(), NN::ParameterRole::Bias);
    EXPECT_EQ(offset.fan_in(), 4);
    EXPECT_EQ(kernel.init().kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_EQ(kernel.role(), NN::ParameterRole::Weight);
    EXPECT_TRUE(graph.value(offset.id_in(graph)).requires_grad);
}

/** @brief Tests that a normal fill keeps its deviation and leaves the other kinds as they were. */
TEST(Unit_Module, ANormalFillKeepsItsDeviation)
{
    constexpr NN::ParameterInit drawn = NN::ParameterInit::normal(0.02);
    constexpr NN::ParameterInit scaled = NN::ParameterInit::fan_scaled();
    constexpr NN::ParameterInit fixed = NN::ParameterInit::constant_value(1.5);

    EXPECT_EQ(drawn.kind, NN::ParameterInit::Kind::Normal);
    EXPECT_NEAR(drawn.deviation, 0.02, K_EXACT_TOLERANCE);
    EXPECT_NEAR(drawn.constant, 0.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(scaled.kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_NEAR(scaled.deviation, 0.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(fixed.kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(fixed.constant, 1.5, K_EXACT_TOLERANCE);
    EXPECT_NEAR(fixed.deviation, 0.0, K_EXACT_TOLERANCE);
}

/** @brief Tests that an Embedding role with no stated fill is fan-scaled, as a weight is. */
TEST(Unit_Module, AnEmbeddingRoleWithoutAFillIsFanScaled)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Parameter table;
    (void)table.declare(builder, Shape{5, 3}, NN::ParameterRole::Embedding, 3, 3, "table");

    EXPECT_EQ(table.role(), NN::ParameterRole::Embedding);
    EXPECT_EQ(table.init().kind, NN::ParameterInit::Kind::FanScaled);
}

/** @brief Tests that a Linear's weight is fan-scaled unless the layer states another fill. */
TEST(Unit_Module, ALinearDeclaresTheFillItStates)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{2, 3}, "x");

    NN::Linear fanned{3, 4};
    NN::Linear drawn{3, 4};
    drawn.weight_init = NN::ParameterInit::normal(0.02);
    (void)fanned.forward(builder, x);
    (void)drawn.forward(builder, x);

    EXPECT_EQ(fanned.weight.init().kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_EQ(drawn.weight.init().kind, NN::ParameterInit::Kind::Normal);
    EXPECT_NEAR(drawn.weight.init().deviation, 0.02, K_EXACT_TOLERANCE);
    EXPECT_EQ(drawn.weight.role(), NN::ParameterRole::Weight);
    EXPECT_EQ(drawn.bias.init().kind, NN::ParameterInit::Kind::Constant);
}

/** @brief Tests that each role's own fill is fan-scaled, zero or one, by the role declared. */
TEST(Unit_Module, ARoleDeclareFillsByItsRole)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    Parameter weight;
    Parameter bias;
    Parameter scale;
    (void)weight.declare(builder, Shape{2, 3}, NN::ParameterRole::Weight, 3, 2, "weight");
    (void)bias.declare(builder, Shape{2}, NN::ParameterRole::Bias, 3, 2, "bias");
    (void)scale.declare(builder, Shape{2}, NN::ParameterRole::Scale, 2, 2, "scale");

    EXPECT_EQ(weight.init().kind, NN::ParameterInit::Kind::FanScaled);
    EXPECT_EQ(bias.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(bias.init().constant, 0.0, K_EXACT_TOLERANCE);
    EXPECT_EQ(scale.role(), NN::ParameterRole::Scale);
    EXPECT_EQ(scale.init().kind, NN::ParameterInit::Kind::Constant);
    EXPECT_NEAR(scale.init().constant, 1.0, K_EXACT_TOLERANCE);
}

namespace
{
    /** @brief Holds one parameter and one buffer, to be listed as a child. */
    struct StatePair
    {
        Parameter scale{};
        NN::Buffer mean{};

        static constexpr auto fields()
        {
            return std::tuple{
                NN::field("scale", &StatePair::scale),
                NN::buffer("mean", &StatePair::mean),
            };
        }

        /** @brief Declares the parameter into @p builder and defines the buffer. */
        void describe(const Graph::GraphBuilder& builder)
        {
            (void)scale.declare(builder, Shape{4}, NN::ParameterRole::Scale, 4, 4, "scale");
            const Status defined = mean.define(Shape{4}, 0.0);
            static_cast<void>(defined);
        }
    };

    /** @brief Lists all three field kinds, with a buffer first and one buffer left undefined. */
    struct MixedState
    {
        NN::Buffer count{};
        StatePair pair{};
        Parameter offset{};
        NN::Buffer unset{};

        static constexpr auto fields()
        {
            return std::tuple{
                NN::buffer("count", &MixedState::count),
                NN::child("pair", &MixedState::pair),
                NN::field("offset", &MixedState::offset),
                NN::buffer("unset", &MixedState::unset),
            };
        }

        /** @brief Describes every piece of state except the buffer named "unset". */
        void describe(const Graph::GraphBuilder& builder)
        {
            const Status defined = count.define(Shape{1}, 0.0, DType::Int64);
            static_cast<void>(defined);
            pair.describe(builder);
            (void)offset.declare(builder, Shape{4}, NN::ParameterRole::Bias, 4, 4, "offset");
        }
    };

    /** @brief Collects what walk_state() yields as "kind name:extents" strings. */
    template<typename M>
    std::vector<std::string> walk_all(M& model, std::string_view prefix)
    {
        std::vector<std::string> seen;
        const auto on_parameter = [&seen](const std::string& name, Parameter& parameter)
        {
            seen.push_back("parameter " + name + ":" + parameter.shape().to_string());
        };
        const auto on_buffer = [&seen](const std::string& name, NN::Buffer& buffer)
        {
            seen.push_back("buffer " + name + ":" + buffer.shape().to_string());
        };

        const NN::StateVisitor visitor(on_parameter, on_buffer);
        NN::walk_state(model, prefix, visitor);
        return seen;
    }

    /** @brief Collects the names for_each_buffer() yields. */
    template<typename M>
    std::vector<std::string> buffer_names(M& model)
    {
        std::vector<std::string> names;
        NN::for_each_buffer(model, [&names](const std::string& name, NN::Buffer&)
        {
            names.push_back(name);
        });
        return names;
    }
}

/** @brief Tests that walk_state() yields all three field kinds in declaration order. */
TEST(Unit_Module, WalkStateYieldsTheThreeFieldKindsInDeclarationOrder)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    model.describe(builder);

    EXPECT_EQ(walk_all(model, ""), (std::vector<std::string>{
        "buffer count:[1]",
        "parameter pair.scale:[4]",
        "buffer pair.mean:[4]",
        "parameter offset:[4]",
    }));
}

/** @brief Tests that walk_state() puts its prefix in front of every name, children included. */
TEST(Unit_Module, WalkStateJoinsItsPrefixToEveryName)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    model.describe(builder);

    EXPECT_EQ(walk_all(model, "model.3"), (std::vector<std::string>{
        "buffer model.3.count:[1]",
        "parameter model.3.pair.scale:[4]",
        "buffer model.3.pair.mean:[4]",
        "parameter model.3.offset:[4]",
    }));
}

/** @brief Tests that for_each_parameter() yields the parameters of the walk and no buffer. */
TEST(Unit_Module, ForEachParameterNeverYieldsABuffer)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    model.describe(builder);

    EXPECT_EQ(walk(model), (std::vector<std::string>{"pair.scale:[4]", "offset:[4]"}));
    EXPECT_EQ(NN::parameter_count(model), 2u);
    EXPECT_EQ(NN::parameter_elements(model), 8);
}

/** @brief Tests that for_each_buffer() yields the buffers of the walk and no parameter. */
TEST(Unit_Module, ForEachBufferNeverYieldsAParameter)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    model.describe(builder);

    EXPECT_EQ(buffer_names(model), (std::vector<std::string>{"count", "pair.mean"}));
    EXPECT_EQ(NN::buffer_count(model), 2u);
}

/** @brief Tests that the walk skips a buffer until define() is called on it. */
TEST(Unit_Module, AnUndefinedBufferIsSkipped)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    EXPECT_TRUE(buffer_names(model).empty());

    model.describe(builder);
    EXPECT_EQ(NN::buffer_count(model), 2u);

    SA_EXPECT_OK(model.unset.define(Shape{2}, 1.0));
    EXPECT_EQ(buffer_names(model), (std::vector<std::string>{"count", "pair.mean", "unset"}));
}

/** @brief Tests that a defined buffer is walked although no graph holds a leaf for it. */
TEST(Unit_Module, ADefinedBufferIsWalkedWithoutALeaf)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    MixedState model;
    model.describe(builder);

    EXPECT_EQ(model.count.id_in(graph), Graph::K_NO_VALUE);
    EXPECT_EQ(NN::buffer_count(model), 2u);
}

namespace
{
    /** @brief Walks its own state in an order no fields() list of it would give. */
    class SelfWalkingStub
    {
        public:
            SelfWalkingStub()
            {
                (void)gain_.declare(builder_, Shape{3}, NN::ParameterRole::Weight, 3, 3, "gain");
                const Status defined = level_.define(Shape{3}, 0.5);
                static_cast<void>(defined);
            }

            void walk_state(std::string_view prefix, const NN::StateVisitor& visitor)
            {
                visitor.buffer(NN::Detail::join(prefix, "level"), level_);
                visitor.parameter(NN::Detail::join(prefix, "gain"), gain_);
            }

        private:
            Graph::Graph graph_{};
            Graph::GraphBuilder builder_{graph_};
            Parameter gain_{};
            NN::Buffer level_{};
    };

    /** @brief Lists a self-walking child and a walk_parameters child by name. */
    struct StubPair
    {
        SelfWalkingStub own{};
        WalkableStub legacy{};

        static constexpr auto fields()
        {
            return std::tuple{
                NN::child("own", &StubPair::own),
                NN::child("legacy", &StubPair::legacy),
            };
        }
    };
}

/** @brief Tests that a type with its own walk_state() is enumerated by it, order included. */
TEST(Unit_Module, ATypesOwnWalkStateIsTheWalk)
{
    static_assert(NN::Detail::HasStateWalk<SelfWalkingStub>::value,
                  "the stub must be detected as walking its own state");
    static_assert(!NN::Detail::HasStateWalk<Linear>::value,
                  "a layer listed by fields() must not take the own-walk arm");

    SelfWalkingStub stub;
    EXPECT_EQ(walk_all(stub, "head"), (std::vector<std::string>{
        "buffer head.level:[3]",
        "parameter head.gain:[3]",
    }));
    EXPECT_EQ(walk(stub), (std::vector<std::string>{"gain:[3]"}));
    EXPECT_EQ(buffer_names(stub), (std::vector<std::string>{"level"}));
}

/** @brief Tests that a child field names a self-walking child and a walk_parameters child. */
TEST(Unit_Module, AChildFieldNamesEveryKindOfWalkableChild)
{
    StubPair pair;
    EXPECT_EQ(walk_all(pair, ""), (std::vector<std::string>{
        "buffer own.level:[3]",
        "parameter own.gain:[3]",
        "parameter legacy.weight:[2, 3]",
        "parameter legacy.bias:[2]",
    }));
}

/** @brief Tests that a walk_parameters type yields its parameters and no buffer. */
TEST(Unit_Module, AWalkParametersTypeYieldsNoBuffer)
{
    WalkableStub stub;
    EXPECT_EQ(walk_all(stub, ""), (std::vector<std::string>{
        "parameter weight:[2, 3]",
        "parameter bias:[2]",
    }));
    EXPECT_EQ(NN::buffer_count(stub), 0u);
}

/** @brief Tests that a LayerAdapter, which walks through walk_parameters, yields its layer. */
TEST(Unit_Module, ALayerAdapterYieldsItsLayersParametersUnderThePrefix)
{
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    NN::LayerAdapter<Linear> adapter(Linear{4, 3}, int64_t{3});
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");
    (void)adapter.forward(builder, x);

    EXPECT_EQ(walk_all(adapter, "2"), (std::vector<std::string>{
        "parameter 2.weight:[3, 4]",
        "parameter 2.bias:[3]",
    }));
    EXPECT_EQ(NN::buffer_count(adapter), 0u);
}

/** @brief Tests that the name check accepts distinct names and refuses a repeated one. */
TEST(Unit_Module, TwoFieldsWithOneNameAreDetected)
{
    // walk_state() asserts the same check, so a layer that repeats a name does not compile.
    static_assert(NN::Detail::has_unique_names(MixedState::fields()),
                  "four distinct names must pass");
    static_assert(NN::Detail::has_unique_names(std::tuple<>{}),
                  "an empty list has no repeated name");
    static_assert(!NN::Detail::has_unique_names(std::tuple{
                      NN::field("weight", &Linear::weight),
                      NN::field("weight", &Linear::bias)}),
                  "two parameters with one name must be refused");
    static_assert(!NN::Detail::has_unique_names(std::tuple{
                      NN::buffer("pair", &MixedState::count),
                      NN::field("offset", &MixedState::offset),
                      NN::child("pair", &MixedState::pair)}),
                  "a buffer and a child with one name must be refused");

    EXPECT_TRUE(NN::Detail::has_unique_names(Linear::fields()));
}

/** @brief Tests that a Sequential of Modules keeps its builder forward and stays a Module. */
TEST(Unit_Module, ASequentialOfModulesKeepsItsBuilderForward)
{
    using Chain = Sequential<Linear, ReLU>;

    static_assert(NN::Module<Chain>, "a Sequential of Modules must stay a Module");
    static_assert(NN::Module<Sequential<Chain, Linear>>,
                  "a Sequential of Modules must stay a Module when it nests");
    static_assert(std::is_same<decltype(std::declval<Chain&>().forward(
                                   std::declval<const Graph::GraphBuilder&>(),
                                   std::declval<Graph::ValueID>())),
                               Graph::ValueID>::value,
                  "the builder forward must return exactly a ValueID");

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    Sequential model{Chain{Linear{4, 8}, ReLU{}}, Linear{8, 3}};
    const Graph::ValueID x = builder.input(Shape{2, 4}, "x");

    EXPECT_EQ(graph.value(model.forward(builder, x)).shape, (Shape{2, 3}));
    EXPECT_EQ(walk(model), (std::vector<std::string>{
        "0.0.weight:[8, 4]", "0.0.bias:[8]",
        "1.weight:[3, 8]", "1.bias:[3]",
    }));
}
