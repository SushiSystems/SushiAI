/**************************************************************************/
/* test_module.cpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the layer library is host-lane, so the parameter walk - the
// thing that decides whether the optimizer sees a layer at all - is testable
// without hardware.

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

// =============================================================================
// docs/TUTORIAL.md §6's custom layer, compiled
// =============================================================================
//
// A tutorial whose code sample does not compile is worse than no tutorial, and
// prose cannot be made to fail a build. So the sample lives here as well, and
// the two must be kept identical: if you edit one, edit the other.
//
// The cases below also pin the two claims §6 makes *about* the sample - that
// bias_add is required rather than a stylistic choice, and that the two ways of
// getting fields() wrong fail differently.

namespace
{
    /** @brief docs/TUTORIAL.md §6, verbatim. A learnable per-feature offset. */
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

    // The role the tutorial gives it, because it is what decides the
    // initialiser: a BIAS starts at zero, a WEIGHT gets a fan-scaled draw.
    EXPECT_EQ(layer.offset.role(), NN::ParameterRole::BIAS);

    // It satisfies the Module contract, so it composes.
    static_assert(NN::Module<MyOffset>, "the tutorial's layer must be a Module");
}

TEST(Unit_Module, TheTutorialIsRightThatMulWouldNotBroadcast)
{
    // §6 says bias_add is required rather than preferred, because mul goes
    // through require_same and there is no implicit broadcast anywhere in this
    // builder. That is the whole reason the sample reads the way it does, so it
    // is asserted rather than left as a remark.
    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);

    const Graph::ValueId x = builder.input(Shape{2, 4}, "x");
    const Graph::ValueId b = builder.input(Shape{4}, "b");

    EXPECT_THROW((void)builder.mul(x, b), Error);
    EXPECT_NO_THROW((void)builder.bias_add(x, b));
}

// ---------------------------------------------------------------------------
// The third arm: a layer whose parameters are enumerated at run time.
//
// The stub below is deliberately not a LayerStack. This file tests the walk,
// and the walk's contract is `walk_parameters(prefix, ParameterVisitor)` -
// anything satisfying it is enumerated through it, whether or not it is one of
// the library's own types. Testing it against LayerStack would tie the walk's
// test to a type that arrived two commits later.
// ---------------------------------------------------------------------------

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
    // The two arms compose: the container supplies "0.", the walkable child
    // supplies the leaf names, and nothing about the join changed.
    StubHolder holder;
    EXPECT_EQ(walk(holder), (std::vector<std::string>{"0.weight:[2, 3]", "0.bias:[2]"}));
}

TEST(Unit_Module, TheRuntimeArmIsAdditiveAndNoExistingTypeTakesIt)
{
    // layers.hpp asserts this for Linear and Sequential at build time; restated
    // here so the property is visible in the test that covers the walk, and so
    // it names the detector rather than the arm's consequences.
    static_assert(NN::Detail::HasParameterWalk<WalkableStub>::value,
                  "the stub must be detected as walking itself");
    static_assert(!NN::Detail::HasParameterWalk<Linear>::value,
                  "a compile-time leaf must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<Sequential<Linear, ReLU>>::value,
                  "a compile-time container must not take the runtime arm");
    static_assert(!NN::Detail::HasParameterWalk<StubHolder>::value,
                  "a container holding a walkable child is still a container");
}
