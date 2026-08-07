/**************************************************************************/
/* test_spatial_shapes.cpp                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

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
            // [N,H,W,C] with a 3x3 stride-1 pad-1 window over 5x5 gives 5x5
            // windows per image, and each row is one window's kernel_h *
            // kernel_w * C taps - channels innermost, so a sub-group reads a
            // contiguous run (CV-3).
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
            // A dilated 3x3 covers 5 pixels, so an unpadded 7 yields 3 windows;
            // stride 2 over the same input yields 2. Both numbers are wrong
            // under the naive (H - K)/S + 1 that forgets dilation.
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
            // Produces a zero extent, which Shape refuses anyway - but the
            // message a caller gets should name the geometry, not the Shape
            // invariant it happened to trip.
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
            // col2im cannot derive its output extents: ConvGeometry carries
            // pad_top and pad_left but no pad_bottom or pad_right, so an
            // asymmetric padding would be silently mis-shaped. The caller
            // states the image shape instead - it always has it, because it
            // held the value im2col consumed.
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
            // A 3x3 stride-1 window with padding only on the bottom/right of a
            // 13x13 input still yields 13x13 windows, but that is NOT what the
            // symmetric formula (13-3)/1+1 = 11 would say. The forward pass
            // (im2col_with_extents) states 13x13 outright; the backward rule
            // must be able to state the same numbers back, because it is the
            // one holding them, not re-deriving them from ConvGeometry alone.
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 13, 13, 3}, "x");

            Graph::ConvGeometry g;
            g.kernel_h = 3; g.kernel_w = 3;
            // pad_top = pad_left = 0: padding was applied bottom/right only,
            // which ConvGeometry cannot express and does not need to.

            const ValueId cols = builder.im2col_with_extents(x, g, 13, 13);
            EXPECT_EQ(graph.values()[cols].shape, Shape({1 * 13 * 13, 3 * 3 * 3}));

            const ValueId image = builder.col2im_with_extents(cols, Shape({1, 13, 13, 3}),
                                                               g, 13, 13);
            EXPECT_EQ(graph.values()[image].shape, Shape({1, 13, 13, 3}));
        }

        TEST(Unit_SpatialShapes, Col2ImDerivingStillRejectsThatSameAsymmetricPair)
        {
            // The point of the twin above: the deriving col2im assumes
            // symmetric padding and is NOT a substitute for it. Feeding it the
            // exact cols/image pair that col2im_with_extents accepted above
            // must still throw, proving the twin is load-bearing rather than
            // decorative.
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
            // im2col reads extents off axes 1,2,3 assuming NHWC. Handed an
            // NCHW value it would read channels as height and produce a
            // plausible but silently wrong shape instead of throwing.
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
            // The indices are a second *result*, not a second output tensor the
            // caller allocates: the backward rule needs exactly the indices
            // this forward produced, and a pair returned together cannot be
            // mismatched the way two separate calls could.
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
            // YOLOv3-tiny's last pool: 2x2 stride 1 over 13x13, padded only on
            // the bottom and right, staying 13x13. ConvGeometry cannot express
            // that padding, so the caller states the extents - and this is the
            // case that proves the explicit overload is load-bearing rather
            // than decorative.
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
            // No indices: its backward divides by a constant window area and
            // needs nothing the forward recorded (CV-6).
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
            // maxpool2d reads extents off axes 1,2,3 assuming NHWC; handed an
            // NCHW value it must throw rather than pool the channel axis.
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
    }
}
