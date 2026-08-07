/**************************************************************************/
/* test_layout.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// Device-free: the IR and its builder are host-lane, which is the whole point
// of keeping shape inference out of the lowering.

#include <gtest/gtest.h>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>

namespace SushiAI
{
    namespace Test
    {
        using Graph::GraphBuilder;
        using Graph::TensorLayout;
        using Graph::ValueId;

        TEST(Unit_Layout, EveryValueIsNhwcUnlessSaidOtherwise)
        {
            // CV-3: one native layout. A value that never mentions layout is
            // NHWC, so a graph built by code that predates this field is
            // uniform by construction rather than by accident.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");

            EXPECT_EQ(graph.values()[x].layout, TensorLayout::NHWC);
        }

        TEST(Unit_Layout, FinaliseAcceptsAUniformGraph)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            (void)builder.matmul(x, w);

            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, FinaliseRejectsAMixedGraphAndNamesTheValue)
        {
            // The failure this exists to catch is a layer emitting an NCHW
            // value into an NHWC network. Nothing downstream would throw: the
            // shapes still line up, every kernel still runs, and the model
            // simply learns from transposed data.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            const ValueId y = builder.matmul(x, w);

            graph.value(y).layout = TensorLayout::NCHW;

            EXPECT_THROW(graph.finalise(), Error);
        }

        TEST(Unit_Layout, LayoutConvertIsAllowedToChangeIt)
        {
            // The one exemption, and the reason the check is per-node rather
            // than a single sweep over every value.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4, 5}, "x");
            const ValueId y = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.values()[y].layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, DeclareLikeInheritsTheSourceValuesLayout)
        {
            // Fourteen tasks still to come each emit a value and carry its
            // layout forward by hand; declare_like is the one place that does
            // it, so a forgotten carry cannot happen.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4, 5}, "x");
            const ValueId y = builder.layout_convert(x, TensorLayout::NCHW);

            const ValueId z = builder.declare_like(y, graph.value(y).shape,
                                                    graph.value(y).dtype,
                                                    Graph::Lifetime::ACTIVATION, "z");

            EXPECT_EQ(graph.values()[z].layout, TensorLayout::NCHW);
        }

        TEST(Unit_Layout, FinaliseStillPropagatesRequiresGrad)
        {
            // The rename must not quietly drop the behaviour the old name
            // described; this is the same assertion test_ir.cpp made.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4}, "x");
            const ValueId w = builder.parameter({4, 3}, "w");
            const ValueId y = builder.matmul(x, w);

            graph.finalise();

            EXPECT_TRUE(graph.values()[y].requires_grad);
        }
    }
}
