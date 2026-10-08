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
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/common.hpp>
#include "../../common/test_throws.hpp"

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            EXPECT_EQ(graph.values()[pooled.indices].dtype, DType::Int64)
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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            const ValueId a_nchw = builder.layout_convert(a, Graph::TensorLayout::Nchw);
            const ValueId b = builder.input({1, 3, 4, 4}, "b");
            const ValueId b_nchw = builder.layout_convert(b, Graph::TensorLayout::Nchw);
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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
            const ValueId gamma = builder.parameter({3}, "gamma");
            const ValueId beta = builder.parameter({3}, "beta");

            EXPECT_THROW((void)builder.channel_affine(x_nchw, gamma, beta), Error);
        }

        /** @brief Tests that batchnorm2d emits four outputs, three of them per channel. */
        TEST(Unit_SpatialShapes, BatchNorm2dKeepsItsShapeAndDeclaresThreePerChannelStatistics)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x", DType::Float64);
            const ValueId gamma = builder.parameter({16}, "gamma", DType::Float64);
            const ValueId beta = builder.parameter({16}, "beta", DType::Float64);

            const Graph::BatchNormResult bn = builder.batchnorm2d(x, gamma, beta, 1e-5);
            EXPECT_EQ(graph.values()[bn.output].shape, Shape({2, 8, 8, 16}));
            EXPECT_EQ(graph.values()[bn.saved_mean].shape, Shape({16}));
            EXPECT_EQ(graph.values()[bn.saved_inv_std].shape, Shape({16}));
            EXPECT_EQ(graph.values()[bn.batch_variance].shape, Shape({16}));

            const Graph::Node& node = graph.node(graph.value(bn.output).producer);
            EXPECT_EQ(node.op, Ops::K_BATCHNORM2D.id);
            EXPECT_EQ(node.outputs, (std::vector<ValueId>{bn.output, bn.saved_mean,
                                                          bn.saved_inv_std, bn.batch_variance}));
            EXPECT_EQ(node.attributes.get(0), 1e-5);

            for (ValueId out : node.outputs)
            {
                EXPECT_EQ(graph.value(out).dtype, DType::Float64) << "value " << out;
                EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation)
                    << "value " << out;
            }

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
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
            const ValueId gamma = builder.parameter({3}, "gamma");
            const ValueId beta = builder.parameter({3}, "beta");

            EXPECT_THROW((void)builder.batchnorm2d(x_nchw, gamma, beta, 1e-5), Error);
        }

        /** @brief Tests that batchnorm_fold emits one node of four operands and two [C] results. */
        TEST(Unit_SpatialShapes, BatchNormFoldYieldsAPerChannelScaleAndShift)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId gamma = builder.parameter({16}, "gamma", DType::Float64);
            const ValueId beta = builder.parameter({16}, "beta", DType::Float64);
            const ValueId mean = builder.buffer({16}, "running_mean", DType::Float64);
            const ValueId variance = builder.buffer({16}, "running_var", DType::Float64);
            const std::size_t nodes_before = graph.node_count();

            const Graph::BatchNormFold fold =
                builder.batchnorm_fold(gamma, beta, mean, variance, 1e-3);

            ASSERT_EQ(graph.node_count(), nodes_before + 1);
            const Graph::Node& node = graph.nodes().back();
            EXPECT_EQ(node.op, Ops::K_BATCHNORM_FOLD.id);
            EXPECT_EQ(node.inputs, (std::vector<ValueId>{gamma, beta, mean, variance}));
            EXPECT_EQ(node.outputs, (std::vector<ValueId>{fold.scale, fold.shift}));
            EXPECT_EQ(node.attributes.get(0), 1e-3);

            for (ValueId out : {fold.scale, fold.shift})
            {
                EXPECT_EQ(graph.value(out).shape, Shape({16})) << "value " << out;
                EXPECT_EQ(graph.value(out).dtype, DType::Float64) << "value " << out;
                EXPECT_EQ(graph.value(out).lifetime, Graph::Lifetime::Activation)
                    << "value " << out;
            }

            // The fold's results are what CHANNEL_AFFINE reads in an evaluation trace.
            const ValueId x = builder.input({2, 8, 8, 16}, "x", DType::Float64);
            EXPECT_EQ(graph.value(builder.channel_affine(x, fold.scale, fold.shift)).shape,
                      Shape({2, 8, 8, 16}));
        }

        /** @brief Tests that batchnorm_fold names the operand whose shape or type does not fit. */
        TEST(Unit_SpatialShapes, BatchNormFoldRejectsAnOperandThatDoesNotMatchGamma)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId gamma = builder.parameter({16}, "gamma");
            const ValueId beta = builder.parameter({16}, "beta");
            const ValueId mean = builder.buffer({16}, "running_mean");
            const ValueId variance = builder.buffer({16}, "running_var");
            const ValueId short_mean = builder.buffer({8}, "short_mean");
            const ValueId wide_variance = builder.buffer({16}, "wide_var", DType::Float64);
            const ValueId matrix = builder.parameter({4, 4}, "matrix");
            const ValueId counts = builder.buffer({16}, "counts", DType::Int64);
            const std::size_t nodes_before = graph.node_count();
            const std::size_t values_before = graph.value_count();

            SA_EXPECT_ERROR((void)builder.batchnorm_fold(gamma, beta, short_mean, variance, 1e-3),
                            "batchnorm_fold", "mean", "[16]", "[8]");
            SA_EXPECT_ERROR((void)builder.batchnorm_fold(gamma, short_mean, mean, variance, 1e-3),
                            "batchnorm_fold", "beta");
            SA_EXPECT_ERROR((void)builder.batchnorm_fold(gamma, beta, mean, wide_variance, 1e-3),
                            "batchnorm_fold", "variance", "float32", "float64");
            SA_EXPECT_ERROR((void)builder.batchnorm_fold(matrix, beta, mean, variance, 1e-3),
                            "batchnorm_fold", "rank-1");
            SA_EXPECT_ERROR((void)builder.batchnorm_fold(counts, counts, counts, counts, 1e-3),
                            "batchnorm_fold", "int64");

            // A rejected fold declares no value and emits no node.
            EXPECT_EQ(graph.node_count(), nodes_before);
            EXPECT_EQ(graph.value_count(), values_before);
        }

        /** @brief Tests that batchnorm_fold accepts a positive epsilon only. */
        TEST(Unit_SpatialShapes, BatchNormFoldRejectsANonPositiveEpsilon)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId gamma = builder.parameter({4}, "gamma");
            const ValueId beta = builder.parameter({4}, "beta");
            const ValueId mean = builder.buffer({4}, "running_mean");
            const ValueId variance = builder.buffer({4}, "running_var");

            SA_EXPECT_ERROR((void)builder.batchnorm_fold(gamma, beta, mean, variance, 0.0),
                            "batchnorm_fold", "epsilon", "positive");
            SA_EXPECT_ERROR((void)builder.batchnorm_fold(gamma, beta, mean, variance, -1e-3),
                            "batchnorm_fold", "epsilon", "positive");
            EXPECT_NO_THROW((void)builder.batchnorm_fold(gamma, beta, mean, variance, 1e-3));
        }

        TEST(Unit_SpatialShapes, SumChannelsFoldsEverythingButTheChannelAxis)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 8, 16}, "x");

            EXPECT_EQ(graph.values()[builder.sum_channels(x)].shape, Shape({16}));

            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
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

        namespace
        {
            /** @brief Returns the operation names of the nodes of @p graph from @p from on. */
            std::vector<std::string> op_names_from(const Graph::Graph& graph, std::size_t from)
            {
                std::vector<std::string> names;
                for (std::size_t i = from; i < graph.node_count(); ++i)
                    names.emplace_back(Ops::name_of(graph.nodes()[i].op));
                return names;
            }

            /** @brief Returns a square unit-stride window of @p kernel taps padded by @p pad. */
            Graph::ConvGeometry square_window(int64_t kernel, int64_t pad)
            {
                Graph::ConvGeometry g;
                g.kernel_h = kernel;
                g.kernel_w = kernel;
                g.pad_top = pad;
                g.pad_left = pad;
                return g;
            }

            /** @brief Returns a unit-stride unpadded window of @p kernel_h by @p kernel_w taps. */
            Graph::ConvGeometry window(int64_t kernel_h, int64_t kernel_w)
            {
                Graph::ConvGeometry g;
                g.kernel_h = kernel_h;
                g.kernel_w = kernel_w;
                return g;
            }
        } // namespace

        /** @brief Tests that conv2d returns an NHWC value with one channel per filter. */
        TEST(Unit_SpatialShapes, Conv2dReturnsAnNhwcValueWithOneChannelPerFilter)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const ValueId bias = builder.parameter({8}, "bias");

            const ValueId y = builder.conv2d(x, weight, bias, square_window(3, 1));

            EXPECT_EQ(graph.values()[y].shape, Shape({2, 5, 5, 8}));
            EXPECT_EQ(graph.values()[y].layout, Graph::TensorLayout::Nhwc);
        }

        /** @brief Tests that conv2d divides the output extents by its stride. */
        TEST(Unit_SpatialShapes, Conv2dDividesTheOutputExtentsByItsStride)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 8, 6, 3}, "x");
            const ValueId weight = builder.parameter({4, 3, 3, 3}, "weight");

            Graph::ConvGeometry g = square_window(3, 1);
            g.stride_h = 2;
            g.stride_w = 2;

            const ValueId y = builder.conv2d(x, weight, Graph::K_NO_VALUE, g);

            EXPECT_EQ(graph.values()[y].shape, Shape({2, 4, 3, 4}));
        }

        /** @brief Tests that a biased conv2d emits RESHAPE, IM2COL, MATMUL, BIAS_ADD, RESHAPE. */
        TEST(Unit_SpatialShapes, Conv2dWithABiasEmitsIm2colMatmulAndBiasAddBetweenTwoReshapes)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const ValueId bias = builder.parameter({8}, "bias");
            const std::size_t operands = graph.node_count();

            (void)builder.conv2d(x, weight, bias, square_window(3, 1));

            EXPECT_EQ(op_names_from(graph, operands), (std::vector<std::string>{
                "sushiai.reshape",
                "sushiai.im2col",
                "sushiai.matmul",
                "sushiai.bias_add",
                "sushiai.reshape",
            }));
        }

        /** @brief Tests that conv2d without a bias emits no BIAS_ADD. */
        TEST(Unit_SpatialShapes, Conv2dWithoutABiasEmitsNoBiasAdd)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const std::size_t operands = graph.node_count();

            (void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(3, 1));

            EXPECT_EQ(op_names_from(graph, operands), (std::vector<std::string>{
                "sushiai.reshape",
                "sushiai.im2col",
                "sushiai.matmul",
                "sushiai.reshape",
            }));
        }

        /** @brief Tests that conv2d hands MATMUL a [rows, taps] and a [C_out, taps] operand. */
        TEST(Unit_SpatialShapes, Conv2dContractsTheColumnMatrixAgainstTheTransposedFilterMatrix)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const std::size_t operands = graph.node_count();

            (void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(3, 1));

            const Graph::Node& matmul = graph.nodes()[operands + 2];
            ASSERT_EQ(matmul.op, Ops::K_MATMUL.id);
            EXPECT_EQ(graph.values()[matmul.inputs[0]].shape, Shape({2 * 5 * 5, 3 * 3 * 3}));
            EXPECT_EQ(graph.values()[matmul.inputs[1]].shape, Shape({8, 3 * 3 * 3}));
            EXPECT_FALSE(matmul.attributes.flag(0));
            EXPECT_TRUE(matmul.attributes.flag(1));
        }

        /** @brief Tests that a 1x1 unit-stride unpadded conv2d emits no IM2COL. */
        TEST(Unit_SpatialShapes, Conv2dPointwiseEmitsNoIm2col)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 1, 1, 3}, "weight");
            const std::size_t operands = graph.node_count();

            const ValueId y = builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(1, 0));

            EXPECT_EQ(op_names_from(graph, operands), (std::vector<std::string>{
                "sushiai.reshape",
                "sushiai.reshape",
                "sushiai.matmul",
                "sushiai.reshape",
            }));
            EXPECT_EQ(graph.values()[y].shape, Shape({2, 5, 5, 8}));
        }

        /** @brief Tests that a 1x1 conv2d with a stride above one still emits IM2COL. */
        TEST(Unit_SpatialShapes, Conv2dStridedOneByOneKernelStillEmitsIm2col)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 6, 6, 3}, "x");
            const ValueId weight = builder.parameter({8, 1, 1, 3}, "weight");
            const std::size_t operands = graph.node_count();

            Graph::ConvGeometry g = square_window(1, 0);
            g.stride_h = 2;
            g.stride_w = 2;

            const ValueId y = builder.conv2d(x, weight, Graph::K_NO_VALUE, g);

            EXPECT_EQ(op_names_from(graph, operands), (std::vector<std::string>{
                "sushiai.reshape",
                "sushiai.im2col",
                "sushiai.matmul",
                "sushiai.reshape",
            }));
            EXPECT_EQ(graph.values()[y].shape, Shape({2, 3, 3, 8}));
        }

        /** @brief Tests that conv2d reads the weight's axes 1 and 2 as kernel height and width. */
        TEST(Unit_SpatialShapes, Conv2dTakesANonSquareKernelInHeightThenWidthOrder)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 1, 3, 3}, "weight");

            const ValueId y = builder.conv2d(x, weight, Graph::K_NO_VALUE, window(1, 3));

            EXPECT_EQ(graph.values()[y].shape, Shape({2, 5, 3, 8}));
        }

        /** @brief Tests that conv2d rejects a weight holding the window's taps transposed. */
        TEST(Unit_SpatialShapes, Conv2dRejectsAWeightWhoseKernelExtentsDifferFromTheWindow)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 1, 3}, "weight");

            // [8,3,1,3] holds as many elements as [8,1,3,3], so no later reshape rejects it.
            EXPECT_THROW((void)builder.conv2d(x, weight, Graph::K_NO_VALUE, window(1, 3)), Error);
        }

        /** @brief Tests that conv2d names itself when a weight's C_in differs from the input's. */
        TEST(Unit_SpatialShapes, Conv2dRejectsAWeightWhoseInputChannelsDifferFromTheImage)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 4}, "weight");

            // reshape rejects the same weight by element count, in a message without "conv2d".
            SA_EXPECT_ERROR(
                (void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(3, 1)), "conv2d");
        }

        /** @brief Tests that conv2d rejects a weight that is not rank 4. */
        TEST(Unit_SpatialShapes, Conv2dRejectsAFlattenedWeight)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3 * 3 * 3}, "weight");

            EXPECT_THROW((void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(3, 1)),
                         Error);
        }

        /** @brief Tests that conv2d rejects a rank-3 input. */
        TEST(Unit_SpatialShapes, Conv2dRejectsARankThreeInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");

            EXPECT_THROW((void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(3, 1)),
                         Error);
        }

        /** @brief Tests that a pointwise conv2d rejects an NCHW input although it skips IM2COL. */
        TEST(Unit_SpatialShapes, Conv2dPointwiseRejectsAnNchwInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId x_nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
            const ValueId weight = builder.parameter({8, 1, 1, 5}, "weight");

            EXPECT_THROW((void)builder.conv2d(x_nchw, weight, Graph::K_NO_VALUE,
                                              square_window(1, 0)),
                         Error);
        }

        /** @brief Tests that conv2d rejects a bias whose extent differs from the filter count. */
        TEST(Unit_SpatialShapes, Conv2dRejectsABiasWhoseExtentDiffersFromTheFilterCount)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const ValueId bias = builder.parameter({4}, "bias");

            EXPECT_THROW((void)builder.conv2d(x, weight, bias, square_window(3, 1)), Error);
        }

        /** @brief Tests that conv2d rejects a window larger than the padded input. */
        TEST(Unit_SpatialShapes, Conv2dRejectsAWindowLargerThanThePaddedInput)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 3}, "x");
            const ValueId weight = builder.parameter({8, 5, 5, 3}, "weight");

            EXPECT_THROW((void)builder.conv2d(x, weight, Graph::K_NO_VALUE, square_window(5, 0)),
                         Error);
        }

        /** @brief Tests that a conv2d call that throws leaves the graph without a new node. */
        TEST(Unit_SpatialShapes, Conv2dEmitsNothingWhenItRejectsItsOperands)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 2, 2, 3}, "x");
            const ValueId fitting = builder.parameter({8, 1, 1, 3}, "fitting");
            const ValueId oversized = builder.parameter({8, 5, 5, 3}, "oversized");
            const ValueId bias = builder.parameter({4}, "bias");
            const ValueId widened = builder.parameter({8, 1, 1, 4}, "widened");
            const std::size_t operands = graph.node_count();

            EXPECT_THROW((void)builder.conv2d(x, oversized, Graph::K_NO_VALUE, square_window(5, 0)),
                         Error);
            EXPECT_THROW((void)builder.conv2d(x, widened, Graph::K_NO_VALUE, square_window(1, 0)),
                         Error);
            EXPECT_THROW((void)builder.conv2d(x, fitting, bias, square_window(1, 0)), Error);

            EXPECT_EQ(graph.node_count(), operands);
        }

        /** @brief Tests that Graph::finalise accepts a graph that conv2d built. */
        TEST(Unit_SpatialShapes, Conv2dBuildsAGraphThatFinaliseAccepts)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const ValueId bias = builder.parameter({8}, "bias");
            const ValueId shortcut = builder.parameter({8, 1, 1, 8}, "shortcut");

            const ValueId y = builder.conv2d(x, weight, bias, square_window(3, 1));
            (void)builder.conv2d(y, shortcut, Graph::K_NO_VALUE, square_window(1, 0));

            EXPECT_NO_THROW(graph.finalise());
        }

        /** @brief Tests that a loss over conv2d differentiates to a weight and a bias gradient. */
        TEST(Unit_SpatialShapes, Conv2dDifferentiatesToAWeightAndABiasGradient)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({2, 5, 5, 3}, "x");
            const ValueId targets = builder.input({2, 8}, "targets");
            const ValueId weight = builder.parameter({8, 3, 3, 3}, "weight");
            const ValueId bias = builder.parameter({8}, "bias");

            const ValueId y = builder.conv2d(x, weight, bias, square_window(3, 1));
            const ValueId logits = builder.spatial_mean(y);
            const ValueId loss = builder.nll_loss(builder.log_softmax(logits), targets);

            const std::array<ValueId, 2> wrt = {weight, bias};
            const std::vector<ValueId> grads = Autograd::differentiate(
                graph, loss, SushiRuntime::span<const ValueId>(wrt.data(), wrt.size()));

            ASSERT_EQ(grads.size(), 2u);
            ASSERT_NE(grads[0], Graph::K_NO_VALUE);
            ASSERT_NE(grads[1], Graph::K_NO_VALUE);
            EXPECT_EQ(graph.values()[grads[0]].shape, Shape({8, 3, 3, 3}));
            EXPECT_EQ(graph.values()[grads[1]].shape, Shape({8}));
        }

        /** @brief Tests that window_extents pads both sides of an axis by its leading pad. */
        TEST(Unit_SpatialShapes, WindowExtentsPadBothSidesOfAnAxisByTheLeadingPad)
        {
            Graph::ConvGeometry g = window(3, 5);
            g.pad_top = 1;
            g.pad_left = 2;

            const Graph::SpatialExtents out = Graph::window_extents(7, 9, g);

            EXPECT_EQ(out.height, 7);
            EXPECT_EQ(out.width, 9);
        }

        /** @brief Tests that window_extents applies the stride and the dilation of each axis. */
        TEST(Unit_SpatialShapes, WindowExtentsHonourStrideAndDilationPerAxis)
        {
            Graph::ConvGeometry g = window(3, 3);
            g.stride_h = 2;
            g.stride_w = 3;
            g.dilation_h = 2;

            const Graph::SpatialExtents out = Graph::window_extents(11, 16, g);

            EXPECT_EQ(out.height, 4);
            EXPECT_EQ(out.width, 5);
        }

        /** @brief Tests that a window as large as the padded input yields one window. */
        TEST(Unit_SpatialShapes, WindowExtentsOfAWindowThatExactlyFitsAreOneByOne)
        {
            const Graph::SpatialExtents out = Graph::window_extents(3, 3, square_window(5, 1));

            EXPECT_EQ(out.height, 1);
            EXPECT_EQ(out.width, 1);
        }

        /** @brief Tests that window_extents names a kernel extent that is not positive. */
        TEST(Unit_SpatialShapes, WindowExtentsRejectANonPositiveKernel)
        {
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, window(0, 3)),
                            "positive kernel", "kernel 0x3");
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, window(3, -2)),
                            "positive kernel", "kernel 3x-2");
        }

        /** @brief Tests that a zero or negative stride is reported as the stride. */
        TEST(Unit_SpatialShapes, WindowExtentsRejectANonPositiveStrideAndNameIt)
        {
            Graph::ConvGeometry zero = window(3, 3);
            zero.stride_h = 0;
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, zero),
                            "positive stride", "stride 0,1");

            Graph::ConvGeometry negative = window(3, 3);
            negative.stride_w = -1;
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, negative),
                            "positive stride", "stride 1,-1");
        }

        /** @brief Tests that window_extents names a dilation that is not positive. */
        TEST(Unit_SpatialShapes, WindowExtentsRejectANonPositiveDilation)
        {
            Graph::ConvGeometry g = window(3, 3);
            g.dilation_w = 0;

            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, g),
                            "positive dilation", "dilation 1,0");
        }

        /** @brief Tests that window_extents refuses a negative padding on either axis. */
        TEST(Unit_SpatialShapes, WindowExtentsRejectANegativePadding)
        {
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, square_window(3, -1)),
                            "padding -1,-1");

            Graph::ConvGeometry left_only = window(3, 3);
            left_only.pad_left = -1;
            SA_EXPECT_ERROR((void)Graph::window_extents(8, 8, left_only), "padding 0,-1");
        }

        /** @brief Tests that window_extents names the window and the input it does not fit. */
        TEST(Unit_SpatialShapes, WindowExtentsRejectAWindowThatDoesNotFit)
        {
            SA_EXPECT_ERROR((void)Graph::window_extents(4, 6, window(5, 5)),
                            "5x5 window", "4x6 input");
        }

        /** @brief Tests that upsampled_extents multiplies each axis by its own scale. */
        TEST(Unit_SpatialShapes, UpsampledExtentsScaleEachAxisByItsOwnFactor)
        {
            const Graph::SpatialExtents out = Graph::upsampled_extents(3, 5, 2, 4);

            EXPECT_EQ(out.height, 6);
            EXPECT_EQ(out.width, 20);
        }

        /** @brief Tests that upsampled_extents refuses a scale below one on either axis. */
        TEST(Unit_SpatialShapes, UpsampledExtentsRejectAScaleBelowOne)
        {
            SA_EXPECT_ERROR((void)Graph::upsampled_extents(3, 5, 0, 2), "at least one", "0x2");
            SA_EXPECT_ERROR((void)Graph::upsampled_extents(3, 5, 2, -1), "at least one", "2x-1");
        }

        /** @brief Tests that upsampled_extents refuses a product beyond 64 bits on either axis. */
        TEST(Unit_SpatialShapes, UpsampledExtentsRejectAProductBeyondSixtyFourBits)
        {
            const int64_t scale = int64_t{1} << 62;

            SA_EXPECT_ERROR((void)Graph::upsampled_extents(5, 1, scale, 1), "fit in 64 bits");
            SA_EXPECT_ERROR((void)Graph::upsampled_extents(1, 5, 1, scale), "fit in 64 bits");
        }

        /** @brief Tests that upsampled_extents accepts the largest product 64 bits hold. */
        TEST(Unit_SpatialShapes, UpsampledExtentsAcceptTheLargestProductThatFits)
        {
            const int64_t largest = std::numeric_limits<int64_t>::max();

            const Graph::SpatialExtents out = Graph::upsampled_extents(largest / 2, 1, 2, 1);

            EXPECT_EQ(out.height, largest - 1);
            EXPECT_EQ(out.width, 1);
        }

        /** @brief Tests that upsample2d refuses an overflowing extent and emits no node. */
        TEST(Unit_SpatialShapes, UpsampleRejectsAScaledExtentBeyondSixtyFourBits)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 1, 5, 3}, "x");
            const std::size_t operands = graph.node_count();

            SA_EXPECT_ERROR((void)builder.upsample2d(x, 1, int64_t{1} << 62), "fit in 64 bits");
            EXPECT_EQ(graph.node_count(), operands);
        }

        /** @brief Tests that the deriving emitters refuse a negative padding by name. */
        TEST(Unit_SpatialShapes, TheDerivingEmittersRejectANegativePadding)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 8, 8, 3}, "x");
            const ValueId weight = builder.parameter({4, 3, 3, 3}, "weight");
            const Graph::ConvGeometry g = square_window(3, -1);

            SA_EXPECT_ERROR((void)builder.im2col(x, g), "padding");
            SA_EXPECT_ERROR((void)builder.conv2d(x, weight, Graph::K_NO_VALUE, g), "padding");
            SA_EXPECT_ERROR((void)builder.maxpool2d(x, g), "padding");
            SA_EXPECT_ERROR((void)builder.avgpool2d(x, g), "padding");
        }

        /** @brief Tests that col2im refuses a negative padding by name. */
        TEST(Unit_SpatialShapes, Col2ImRejectsANegativePadding)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({36, 27}, "cols");

            SA_EXPECT_ERROR((void)builder.col2im(cols, Shape({1, 8, 8, 3}), square_window(3, -1)),
                            "padding");
        }

        /** @brief Tests that col2im refuses an image shape that is not rank 4. */
        TEST(Unit_SpatialShapes, Col2ImRejectsAnImageShapeThatIsNotRankFour)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId cols = builder.input({9, 27}, "cols");

            SA_EXPECT_ERROR((void)builder.col2im(cols, Shape({5, 5, 3}), square_window(3, 0)),
                            "col2im", "rank-4");
        }

        /** @brief Tests that the deriving emitters name the layout before a window's fit. */
        TEST(Unit_SpatialShapes, TheDerivingEmittersNameTheLayoutBeforeTheWindow)
        {
            Graph::Graph graph;
            const GraphBuilder builder(graph);
            const ValueId x = builder.input({1, 8, 8, 2}, "x");
            const ValueId nchw = builder.layout_convert(x, Graph::TensorLayout::Nchw);
            const Graph::ConvGeometry g = square_window(3, 0);

            SA_EXPECT_ERROR((void)builder.im2col(nchw, g), "im2col", "NHWC");
            SA_EXPECT_ERROR((void)builder.maxpool2d(nchw, g), "maxpool2d", "NHWC");
            SA_EXPECT_ERROR((void)builder.avgpool2d(nchw, g), "avgpool2d", "NHWC");
        }
    }
}
