/**************************************************************************/
/* test_spatial_shapes.cpp                                                */
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
 * @file test_spatial_shapes.cpp
 * @brief Tests the shapes the builder infers for spatial operations and the layouts it refuses.
 * @author Mustafa Garip
 */

#include <array>
#include <gtest/gtest.h>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Test
    {
        using Graph::GraphBuilder;
        using Graph::ValueId;

        TEST(Unit_SpatialShapes, Im2ColProducesTheColumnMatrixAGemmExpects)
        {
            // A row is one window, channels innermost (CV-3); see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId cols = builder.im2col(x, g);

            EXPECT_EQ(graph.values()[cols].shape, Shape({2 * 5 * 5, 3 * 3 * 3}));
        }

        TEST(Unit_SpatialShapes, Im2ColHonoursStrideAndDilation)
        {
            // A dilated 3x3 covers 5 pixels; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 7, 7, 2}, "x");

            Graph::ConvGeometry dilated;
            dilated.kernel_h = 3; dilated.kernel_w = 3;
            dilated.dilation_h = 2; dilated.dilation_w = 2;
            EXPECT_EQ(graph.values()[builder.im2col(x, dilated)].shape,
                      Shape({1 * 3 * 3, 3 * 3 * 2}));

            Graph::ConvGeometry strided;
            strided.kernel_h = 3; strided.kernel_w = 3;
            strided.stride_h = 2; strided.stride_w = 2;
            EXPECT_EQ(graph.values()[builder.im2col(x, strided)].shape,
                      Shape({1 * 3 * 3, 3 * 3 * 2}));
        }

        TEST(Unit_SpatialShapes, Im2ColRejectsAWindowLargerThanThePaddedInput)
        {
            // The message names the geometry, not the Shape; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 1}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 5; g.kernel_w = 5;

            EXPECT_THROW((void)builder.im2col(x, g), Error);
        }

        TEST(Unit_SpatialShapes, Im2ColRejectsANonSpatialInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({4, 8}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 1; g.kernel_w = 1;

            EXPECT_THROW((void)builder.im2col(x, g), Error);
        }

        TEST(Unit_SpatialShapes, Col2ImReturnsToTheImageItWasAskedFor)
        {
            // The caller states the image shape; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({2 * 5 * 5, 3 * 3 * 3}, "cols");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            const ValueId image = builder.col2im(cols, Shape({2, 5, 5, 3}), g);

            EXPECT_EQ(graph.values()[image].shape, Shape({2, 5, 5, 3}));
        }

        TEST(Unit_SpatialShapes, Col2ImRejectsAColumnMatrixThatDoesNotMatchTheImage)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({2 * 5 * 5, 3 * 3 * 3}, "cols");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            EXPECT_THROW((void)builder.col2im(cols, Shape({2, 6, 6, 3}), g), Error);
        }

        TEST(Unit_SpatialShapes, Col2ImWithExtentsRoundTripsAnAsymmetricallyPaddedForward)
        {
            // The backward rule states the 13x13 extents back; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 13, 13, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            // Padding is applied bottom and right only; see tests/functional/unit/README.md.

            const ValueId cols = builder.im2col_with_extents(x, g, 13, 13);
            EXPECT_EQ(graph.values()[cols].shape, Shape({1 * 13 * 13, 3 * 3 * 3}));

            const ValueId image = builder.col2im_with_extents(cols, Shape({1, 13, 13, 3}),
                                                               g, 13, 13);
            EXPECT_EQ(graph.values()[image].shape, Shape({1, 13, 13, 3}));
        }

        TEST(Unit_SpatialShapes, Col2ImDerivingStillRejectsThatSameAsymmetricPair)
        {
            // The deriving col2im assumes symmetric padding; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 13, 13, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;

            const ValueId cols = builder.im2col_with_extents(x, g, 13, 13);

            EXPECT_THROW((void)builder.col2im(cols, Shape({1, 13, 13, 3}), g), Error);
        }

        TEST(Unit_SpatialShapes, Im2ColRejectsAnNchwInput)
        {
            // im2col reads axes 1, 2, 3 as NHWC; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 32, 32}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.pad_top = 1;  g.pad_left = 1;

            EXPECT_THROW((void)builder.im2col(x_nchw, g), Error);
        }

        TEST(Unit_SpatialShapes, MaxPoolProducesAValueAndItsArgmaxTogether)
        {
            // The indices are a second result of the node; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 4, 4, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 2; g.kernel_w = 2;
            g.stride_h = 2; g.stride_w = 2;

            const Graph::PoolResult pooled = builder.maxpool2d(x, g);

            EXPECT_EQ(graph.values()[pooled.output].shape, Shape({2, 2, 2, 3}));
            EXPECT_EQ(graph.values()[pooled.indices].shape, Shape({2, 2, 2, 3}));
            EXPECT_EQ(graph.values()[pooled.indices].dtype, DType::INT64)
                << "an index is not a number to interpolate; a float index would "
                   "compare unequal to the integer the forward wrote";
        }

        TEST(Unit_SpatialShapes, MaxPoolHandlesYoloTinysAsymmetricThirteenToThirteen)
        {
            // The caller states what ConvGeometry cannot; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 13, 13, 256}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 2; g.kernel_w = 2;

            const Graph::PoolResult pooled = builder.maxpool2d_with_extents(x, g, 13, 13);
            EXPECT_EQ(graph.values()[pooled.output].shape, Shape({1, 13, 13, 256}));

            // Without the explicit extents the same geometry gives 12x12.
            EXPECT_EQ(graph.values()[builder.maxpool2d(x, g).output].shape,
                      Shape({1, 12, 12, 256}));
        }

        TEST(Unit_SpatialShapes, AvgPoolProducesOnlyAValue)
        {
            // avgpool2d records no indices (CV-6); see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 6, 6, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.stride_h = 3; g.stride_w = 3;

            EXPECT_EQ(graph.values()[builder.avgpool2d(x, g)].shape, Shape({2, 2, 2, 3}));
        }

        TEST(Unit_SpatialShapes, UpsampleMultipliesTheSpatialExtentsOnly)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 5, 7}, "x");

            EXPECT_EQ(graph.values()[builder.upsample2d(x, 2, 2)].shape, Shape({2, 6, 10, 7}));
        }

        TEST(Unit_SpatialShapes, UpsampleRejectsANonPositiveScale)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 1}, "x");

            EXPECT_THROW((void)builder.upsample2d(x, 0, 2), Error);
        }

        TEST(Unit_SpatialShapes, MaxPoolRejectsAnNchwInput)
        {
            // maxpool2d reads axes 1, 2, 3 as NHWC; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 4, 4}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            Graph::ConvGeometry g;
            g.kernel_h = 2; g.kernel_w = 2;
            g.stride_h = 2; g.stride_w = 2;

            EXPECT_THROW((void)builder.maxpool2d(x_nchw, g), Error);
        }

        TEST(Unit_SpatialShapes, AvgPoolRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 6, 6}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            g.stride_h = 3; g.stride_w = 3;

            EXPECT_THROW((void)builder.avgpool2d(x_nchw, g), Error);
        }

        TEST(Unit_SpatialShapes, UpsampleRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 2, 2}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            EXPECT_THROW((void)builder.upsample2d(x_nchw, 2, 2), Error);
        }

        TEST(Unit_SpatialShapes, SpatialMeanReducesToNAndC)
        {
            // The numeric probe's loss needs [N, C]; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({4, 7, 7, 512}, "x");

            EXPECT_EQ(graph.values()[builder.spatial_mean(x)].shape, Shape({4, 512}));
        }

        TEST(Unit_SpatialShapes, SpatialMeanRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 4, 4}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            EXPECT_THROW((void)builder.spatial_mean(x_nchw), Error);
        }

        TEST(Unit_SpatialShapes, ConcatSumsTheChannelAxisAndNothingElse)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId a = builder.input({2, 8, 8, 16}, "a");
            const ValueId b = builder.input({2, 8, 8, 32}, "b");
            const std::array<ValueId, 2> parts = {a, b};

            EXPECT_EQ(graph.values()[builder.concat_channels(
                          SushiRuntime::span<const ValueId>(parts.data(), parts.size()))].shape,
                      Shape({2, 8, 8, 48}));
        }

        TEST(Unit_SpatialShapes, ConcatRejectsOperandsThatDisagreeOffTheChannelAxis)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId a = builder.input({2, 8, 8, 16}, "a");
            const ValueId b = builder.input({2, 4, 8, 16}, "b");
            const std::array<ValueId, 2> parts = {a, b};

            EXPECT_THROW((void)builder.concat_channels(
                SushiRuntime::span<const ValueId>(parts.data(), parts.size())), Error);
        }

        TEST(Unit_SpatialShapes, ConcatRejectsAnNchwOperand)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId a = builder.input({1, 3, 4, 4}, "a");
            const ValueId a_nchw = builder.layout_convert(a, Graph::TensorLayout::NCHW);
            const ValueId b = builder.input({1, 3, 4, 4}, "b");
            const ValueId b_nchw = builder.layout_convert(b, Graph::TensorLayout::NCHW);
            const std::array<ValueId, 2> parts = {a_nchw, b_nchw};

            EXPECT_THROW((void)builder.concat_channels(
                SushiRuntime::span<const ValueId>(parts.data(), parts.size())), Error);
        }

        TEST(Unit_SpatialShapes, SliceTakesAChannelRangeAndRefusesToOverrun)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 48}, "x");

            EXPECT_EQ(graph.values()[builder.slice_channels(x, 16, 32)].shape,
                      Shape({2, 8, 8, 32}));
            EXPECT_THROW((void)builder.slice_channels(x, 32, 32), Error);
        }

        TEST(Unit_SpatialShapes, SliceRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 4, 4}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);

            EXPECT_THROW((void)builder.slice_channels(x_nchw, 0, 1), Error);
        }

        TEST(Unit_SpatialShapes, ChannelAffineKeepsTheShapeAndChecksTheParameters)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x");
            const ValueId gamma = builder.parameter({16}, "gamma");
            const ValueId beta = builder.parameter({16}, "beta");

            EXPECT_EQ(graph.values()[builder.channel_affine(x, gamma, beta)].shape,
                      Shape({2, 8, 8, 16}));

            const ValueId wrong = builder.parameter({8}, "wrong");
            EXPECT_THROW((void)builder.channel_affine(x, wrong, beta), Error);
        }

        TEST(Unit_SpatialShapes, ChannelAffineRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 4, 4}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
            const ValueId gamma = builder.parameter({3}, "gamma");
            const ValueId beta = builder.parameter({3}, "beta");

            EXPECT_THROW((void)builder.channel_affine(x_nchw, gamma, beta), Error);
        }

        TEST(Unit_SpatialShapes, BatchNorm2dKeepsItsShapeAndDeclaresTwoPerChannelStatistics)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x");
            const ValueId gamma = builder.parameter({16}, "gamma");
            const ValueId beta = builder.parameter({16}, "beta");

            const Graph::BatchNormResult bn = builder.batchnorm2d(x, gamma, beta, 1e-5);
            EXPECT_EQ(graph.values()[bn.output].shape, Shape({2, 8, 8, 16}));
            EXPECT_EQ(graph.values()[bn.saved_mean].shape, Shape({16}));
            EXPECT_EQ(graph.values()[bn.saved_inv_std].shape, Shape({16}));

            const ValueId wrong = builder.parameter({8}, "wrong");
            EXPECT_THROW((void)builder.batchnorm2d(x, wrong, beta, 1e-5), Error);

            // Epsilon must be positive; see tests/functional/unit/README.md.
            EXPECT_THROW((void)builder.batchnorm2d(x, gamma, beta, 0.0), Error);
        }

        TEST(Unit_SpatialShapes, BatchNorm2dRejectsAnNchwInput)
        {
            // The layout check runs before any extent is read; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 3, 4, 4}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
            const ValueId gamma = builder.parameter({3}, "gamma");
            const ValueId beta = builder.parameter({3}, "beta");

            EXPECT_THROW((void)builder.batchnorm2d(x_nchw, gamma, beta, 1e-5), Error);
        }

        TEST(Unit_SpatialShapes, SumChannelsFoldsEverythingButTheChannelAxis)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x");

            EXPECT_EQ(graph.values()[builder.sum_channels(x)].shape, Shape({16}));

            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::NCHW);
            EXPECT_THROW((void)builder.sum_channels(x_nchw), Error);
        }

        TEST(Unit_SpatialShapes, ReshapeRefusesToChangeTheElementCount)
        {
            // A reshape moves no data; see tests/functional/unit/README.md.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 3, 4}, "x");

            EXPECT_EQ(graph.values()[builder.reshape(x, Shape({6, 4}))].shape, Shape({6, 4}));
            EXPECT_THROW((void)builder.reshape(x, Shape({6, 5})), Error);
        }
    }
}
