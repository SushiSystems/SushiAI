/****************************************************************/
/* test_layout.cpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Device-free: the IR and its builder are host-lane, which is the whole point
// of keeping shape inference out of the lowering.

#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/autograd/backward.hpp>
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

        TEST(Unit_Layout, TheThreeRulesThatEmitZerosCannotBeReachedWithAnNchwOperand)
        {
            // Why the zeros exposure is latent rather than live, checked
            // rather than asserted in a comment - because it is the premise
            // the zeros_like() change rests on, and it is a premise about
            // *other* emitters' guards, which is exactly the kind that stops
            // being true without anyone editing this file.
            //
            // slice_channels_backward and channel_affine_backward are the
            // rank-4 sites, and CV-3 makes their forward emitters refuse an
            // NCHW operand outright, so their dy can never carry that tag.
            Graph::Graph graph;
            const GraphBuilder builder(graph);

            const ValueId x = builder.input({1, 2, 2, 2}, "x");
            const ValueId nchw = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_THROW((void)builder.slice_channels(nchw, 0, 1), Error);
            EXPECT_THROW((void)builder.channel_affine(nchw, x, x), Error);

            // sum_rows_backward is the rank-2 site, and reshape carries an
            // NCHW tag into rank-2 land with no guard at all - so unlike the
            // two above, the emitters let this chain be built.
            //
            // What blocks it is the far end. differentiate() only seeds a loss
            // produced by NLL_LOSS, whose one-hot operand is a leaf and so
            // always NHWC, and nothing but LAYOUT_CONVERT may cross back -
            // which is rank-4 only. finalise() rejects that node before
            // differentiation ever runs, so the rule is never reached with an
            // NCHW gradient. Note where the guard actually lives: it is the
            // pre-existing proof, not a check in any emitter, which is why
            // this is worth a test rather than a sentence.
            const ValueId rows = builder.reshape(nchw, Shape{1, 8});
            EXPECT_EQ(graph.value(rows).layout, TensorLayout::NCHW);

            const ValueId target = builder.input({1, 8}, "target");
            EXPECT_EQ(graph.value(target).layout, TensorLayout::NHWC);

            const ValueId summed = builder.sum_rows(rows);
            EXPECT_EQ(graph.value(summed).layout, TensorLayout::NCHW);

            (void)builder.nll_loss(builder.log_softmax(rows), target);
            EXPECT_THROW(graph.finalise(), Error);
        }

        TEST(Unit_Layout, ZerosLikeCarriesTheTagAndPlainZerosCannot)
        {
            // The contract the three backward rules now depend on. ZERO reads
            // nothing, so plain zeros() has no operand to inherit from and its
            // NHWC tag is a default rather than an answer; that is harmless
            // for the ZERO node itself, which the per-node proof skips for
            // having no inputs, and it is the consumer reading those zeros
            // beside a real operand that would see a node crossing layouts.
            Graph::Graph graph;
            const GraphBuilder builder(graph);

            const ValueId x = builder.input({1, 2, 2, 2}, "x");
            const ValueId nchw = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.value(builder.zeros(Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NHWC);
            EXPECT_EQ(graph.value(builder.zeros_like(nchw, Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NCHW);
            EXPECT_EQ(graph.value(builder.zeros_like(x, Shape{1, 2, 2, 2})).layout,
                      TensorLayout::NHWC);
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
