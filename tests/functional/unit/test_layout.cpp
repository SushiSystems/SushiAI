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
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>

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

        namespace
        {
            /**
             * @brief A graph that crosses layouts, with the NCHW side named.
             *
             * Both rebuilding passes copy the value table entry by entry, so
             * what is under test is that copy - not the pass's own subject.
             * The chain is deliberately boring for the same reason: a fusion
             * candidate or a precision decision here would only obscure which
             * field went missing.
             */
            struct Crossed
            {
                Graph::Graph graph;
                ValueId nhwc = Graph::NO_VALUE;
                ValueId nchw = Graph::NO_VALUE;
            };

            Crossed make_crossed()
            {
                Crossed c;
                const GraphBuilder builder(c.graph);

                c.nhwc = builder.input({2, 3, 4, 5}, "x");
                c.nchw = builder.layout_convert(c.nhwc, TensorLayout::NCHW);
                c.graph.finalise();

                return c;
            }
        }

        TEST(Unit_Layout, FusionCarriesTheLayoutTagAcrossItsRebuild)
        {
            // fuse() rewrites the node list and copies the value table. A
            // field it forgets to copy is reset to add_value's default, which
            // is NHWC - so a dropped tag is not a missing layout but a wrong
            // one, and every shape check downstream still agrees with it.
            //
            // What it costs: Lowering::record_layout_convert decides whether
            // to relabel a side into [N,H,W,C] by reading these two tags. A
            // genuine NCHW -> NHWC convert that reads as NHWC -> NHWC applies
            // no transpose and copies transposed data straight through.
            Crossed c = make_crossed();

            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "fuse() reset a NCHW value to NHWC; the convert that produced it is now "
                   "a no-op that copies transposed data through";
        }

        TEST(Unit_Layout, CastInsertionCarriesTheLayoutTagAcrossItsRebuild)
        {
            // The same omission in the other rebuilding pass. Trainer runs
            // both unconditionally, so either one is enough on the default
            // path.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "insert_casts() reset a NCHW value to NHWC";
        }

        TEST(Unit_Layout, ARebuiltGraphStillProvesUniform)
        {
            // The consequence worth stating separately: with every tag reset
            // to one value, finalise() passes for the wrong reason. A graph
            // that is uniform because it was flattened is indistinguishable
            // from one that was uniform to begin with, so the proof has to be
            // re-run against tags that survived.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});
            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(c.graph.finalise());
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
