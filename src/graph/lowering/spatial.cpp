/**************************************************************************/
/* spatial.cpp                                                            */
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
 * @file spatial.cpp
 * @brief Implements the lowering rows of the spatial operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/spatial.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/engine/math/reductions.hpp>
#include <SushiBLAS/engine/spatial.hpp>
#include <SushiBLAS/support/spatial_geometry.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Reads a node's eight geometry slots back into a SushiBLAS geometry.
             * @param node A node whose attribute slots 0 to 7 hold the window description.
             * @return The sliding-window geometry the node describes.
             */
            SushiBLAS::Spatial::ConvGeometry blas_geometry_of(const Node& node) noexcept
            {
                SushiBLAS::Spatial::ConvGeometry g;
                g.kernel_h   = static_cast<int64_t>(node.attributes.get(0));
                g.kernel_w   = static_cast<int64_t>(node.attributes.get(1));
                g.stride_h   = static_cast<int64_t>(node.attributes.get(2));
                g.stride_w   = static_cast<int64_t>(node.attributes.get(3));
                g.dilation_h = static_cast<int64_t>(node.attributes.get(4));
                g.dilation_w = static_cast<int64_t>(node.attributes.get(5));
                g.pad_top    = static_cast<int64_t>(node.attributes.get(6));
                g.pad_left   = static_cast<int64_t>(node.attributes.get(7));
                return g;
            }

            /**
             * @brief Reads the output window grid from attribute slots 8 and 9.
             * @param context The node to lower; both slots must hold a positive extent.
             * @param what The operation's name, as the failure message spells it.
             * @param out_h Receives the output height.
             * @param out_w Receives the output width.
             */
            void windowed_extents_of(const LoweringContext& context, const char* what,
                                     int64_t& out_h, int64_t& out_w)
            {
                const Node& node = context.node();
                out_h = static_cast<int64_t>(node.attributes.get(8));
                out_w = static_cast<int64_t>(node.attributes.get(9));

                SA_FATAL_IF(out_h <= 0 || out_w <= 0,
                            "{} (node {}) carries a {}x{} window grid; its emitter did "
                            "not write the output extents into attribute slots 8 and 9",
                            what, context.id(), out_h, out_w);
            }
        } // namespace

        Status lower_im2col(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());
            int64_t out_h = 0;
            int64_t out_w = 0;
            windowed_extents_of(context, "IM2COL", out_h, out_w);

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& cols, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.im2col(in, cols, g, out_h, out_w);
            context.count(1);
            return Status();
        }

        Status lower_col2im(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());
            int64_t out_h = 0;
            int64_t out_w = 0;
            windowed_extents_of(context, "COL2IM", out_h, out_w);

            const SushiBLAS::Tensor& cols = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& image, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.col2im(cols, image, g, out_h, out_w);
            context.count(1);
            return Status();
        }

        Status lower_maxpool2d(const LoweringContext& context)
        {
            context.require_arity(1, 2);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& indices, context.output(1));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.pool_max(in, out, indices, g);
            context.count(1);
            return Status();
        }

        Status lower_maxpool2d_backward(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());

            const SushiBLAS::Tensor& dy = context.input(0);
            const SushiBLAS::Tensor& indices = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.pool_max_backward(dy, indices, dx, g);
            context.count(1);
            return Status();
        }

        Status lower_avgpool2d(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.pool_avg(in, out, g);
            context.count(1);
            return Status();
        }

        Status lower_avgpool2d_backward(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Spatial::ConvGeometry g = blas_geometry_of(context.node());

            const SushiBLAS::Tensor& dy = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.pool_avg_backward(dy, dx, g);
            context.count(1);
            return Status();
        }

        Status lower_upsample2d(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Node& node = context.node();
            const int64_t scale_h = static_cast<int64_t>(node.attributes.get(0));
            const int64_t scale_w = static_cast<int64_t>(node.attributes.get(1));

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.upsample_nearest(in, out, scale_h, scale_w);
            context.count(1);
            return Status();
        }

        Status lower_upsample2d_backward(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Node& node = context.node();
            const int64_t scale_h = static_cast<int64_t>(node.attributes.get(0));
            const int64_t scale_w = static_cast<int64_t>(node.attributes.get(1));

            const SushiBLAS::Tensor& dy = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.upsample_nearest_backward(dy, dx, scale_h, scale_w);
            context.count(1);
            return Status();
        }

        Status lower_spatial_mean(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::ReductionOps reductions(context.recorder());
            reductions.mean_spatial(in, out);
            context.count(1);
            return Status();
        }

        Status lower_spatial_mean_backward(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Shape& dx_shape = context.output_value(0).shape;
            SA_FATAL_IF(dx_shape.rank() != 4,
                        "SPATIAL_MEAN_BACKWARD (node {}) needs a rank-4 [N,H,W,C] result to read "
                        "H and W from; got {}", context.id(), dx_shape.to_string());

            const double scale = 1.0 / static_cast<double>(dx_shape[1] * dx_shape[2]);

            const SushiBLAS::Tensor& dy = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.broadcast_spatial(dy, dx, scale);
            context.count(1);
            return Status();
        }

        Status lower_concat_channels(const LoweringContext& context)
        {
            context.require_arity_at_least(1, 1);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());

            int64_t offset = 0;
            const std::size_t operands = context.node().inputs.size();
            for (std::size_t i = 0; i < operands; ++i)
            {
                spatial.concat_channels(context.input(i), out, offset);
                offset += context.input_value(i).shape[3];
            }

            // One copy per operand, not one for the node.
            context.count(operands);
            return Status();
        }

        Status lower_slice_channels(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const int64_t offset = static_cast<int64_t>(context.node().attributes.get(0));

            const SushiBLAS::Tensor& in = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.slice_channels(in, offset, out);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
