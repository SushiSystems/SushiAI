/**************************************************************************/
/* test_layout.cpp                                                        */
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
 * @file test_layout.cpp
 * @brief Tests that every value carries a tensor layout and that the rebuilding passes keep it.
 * @author Mustafa Garip
 */

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
            // A value that states no layout is NHWC; see tests/functional/unit/README.md.
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
            // A mixed graph fails finalise(); see tests/functional/unit/README.md.
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
            // LAYOUT_CONVERT alone may change the layout; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4, 5}, "x");
            const ValueId y = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_EQ(graph.values()[y].layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(graph.finalise());
        }

        TEST(Unit_Layout, DeclareLikeInheritsTheSourceValuesLayout)
        {
            // declare_like carries the layout forward; see tests/functional/unit/README.md.
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
             * @brief Holds a graph that crosses layouts, with the NCHW side named.
             * @see tests/functional/unit/README.md
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
            // fuse() copies the layout tag of every value; see tests/functional/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "fuse() reset a NCHW value to NHWC; the convert that produced it is now "
                   "a no-op that copies transposed data through";
        }

        TEST(Unit_Layout, CastInsertionCarriesTheLayoutTagAcrossItsRebuild)
        {
            // Cast insertion copies the layout tag as well; see tests/functional/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});

            EXPECT_EQ(c.graph.value(c.nhwc).layout, TensorLayout::NHWC);
            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW)
                << "insert_casts() reset a NCHW value to NHWC";
        }

        TEST(Unit_Layout, ARebuiltGraphStillProvesUniform)
        {
            // The proof is re-run on tags that survived; see tests/functional/unit/README.md.
            Crossed c = make_crossed();

            (void)Graph::insert_casts(c.graph, Graph::CastOptions{});
            (void)Graph::fuse(c.graph, Graph::FusionOptions{});

            EXPECT_EQ(c.graph.value(c.nchw).layout, TensorLayout::NCHW);
            EXPECT_NO_THROW(c.graph.finalise());
        }

        TEST(Unit_Layout, TheThreeRulesThatEmitZerosCannotBeReachedWithAnNchwOperand)
        {
            // Rank-4 emitters refuse an NCHW operand (CV-3); see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);

            const ValueId x = builder.input({1, 2, 2, 2}, "x");
            const ValueId nchw = builder.layout_convert(x, TensorLayout::NCHW);

            EXPECT_THROW((void)builder.slice_channels(nchw, 0, 1), Error);
            EXPECT_THROW((void)builder.channel_affine(nchw, x, x), Error);

            // finalise() rejects the node ahead of the rule; see tests/functional/unit/README.md.
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
            // zeros_like() inherits its operand's layout; see tests/functional/unit/README.md.
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
            // finalise() still propagates requires_grad; see tests/functional/unit/README.md.
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
