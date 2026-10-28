/**************************************************************************/
/* norm.cpp                                                               */
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
 * @file norm.cpp
 * @brief Implements the lowering rows of the normalization operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/norm.hpp>

#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/random/random.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiBLAS/spatial/spatial.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /**
             * @brief Returns a rank-2 `[N*H*W, C]` view over the storage of an NHWC tensor.
             * @param t The NHWC tensor; must be contiguous.
             * @param rows The product of the batch and the two spatial extents.
             * @param cols The channel count.
             */
            SushiBLAS::Tensor channel_rows(const SushiBLAS::Tensor& t, int64_t rows, int64_t cols)
            {
                const int64_t extents[2] = {rows, cols};
                return SushiBLAS::Tensor(
                    t.storage, SushiRuntime::span<const int64_t>(extents, 2),
                    t.dtype, t.storage_offset, t.layout);
            }

            /**
             * @brief Returns a rank-4 `[rows, 1, 1, cols]` view whose channel axis is a column.
             * @param t A contiguous tensor of rows * cols elements.
             */
            SushiBLAS::Tensor column_channels(const SushiBLAS::Tensor& t, int64_t rows,
                                              int64_t cols)
            {
                const int64_t extents[4] = {rows, 1, 1, cols};
                return SushiBLAS::Tensor(
                    t.storage, SushiRuntime::span<const int64_t>(extents, 4),
                    t.dtype, t.storage_offset, t.layout);
            }

            /**
             * @brief Returns a strided rank-4 `[1, 1, cols, rows]` view whose channel axis is a
             *        row, so a per-channel kernel broadcasts one value over each row.
             * @param t A contiguous tensor of rows * cols elements.
             */
            SushiBLAS::Tensor row_channels(const SushiBLAS::Tensor& t, int64_t rows, int64_t cols)
            {
                const int64_t extents[4] = {1, 1, rows, cols};
                return SushiBLAS::Tensor(
                    t.storage, SushiRuntime::span<const int64_t>(extents, 4),
                    t.dtype, t.storage_offset, t.layout).transpose(2, 3);
            }

            /** @brief Holds the row count and the row length a [..., D] operand is read as. */
            struct RowExtents
            {
                int64_t rows = 0;  /**< @brief The product of every extent but the last. */
                int64_t width = 0; /**< @brief The last extent, D. */
            };

            /** @brief Returns the rows of @p shape; a shape with no axis or no element is fatal. */
            RowExtents row_extents(const LoweringContext& context, const Shape& shape)
            {
                SA_FATAL_IF(shape.rank() < 1 || shape.num_elements() < 1,
                            "{} (node {}) needs a [..., D] operand with an element; got {}",
                            context.name(), context.id(), shape.to_string());

                const int64_t width = shape[shape.rank() - 1];
                return RowExtents{shape.num_elements() / width, width};
            }
        } // namespace

        void request_batchnorm2d_scratch(const Graph& graph, NodeID node,
                                         std::vector<ScratchRequest>& requests) noexcept
        {
            const Value& x = graph.value(graph.node(node).inputs[0]);
            const Shape per_channel({x.shape[3]});

            requests.push_back(ScratchRequest{node, 0, per_channel, x.dtype});
            requests.push_back(ScratchRequest{node, 1, x.shape, x.dtype});
            requests.push_back(ScratchRequest{node, 2, per_channel, x.dtype});
        }

        void request_batchnorm2d_backward_scratch(const Graph& graph, NodeID node,
                                                  std::vector<ScratchRequest>& requests) noexcept
        {
            const Value& x = graph.value(graph.node(node).inputs[1]);
            const Shape per_channel({x.shape[3]});

            requests.push_back(ScratchRequest{node, 0, per_channel, x.dtype});
            requests.push_back(ScratchRequest{node, 1, x.shape, x.dtype});
            requests.push_back(ScratchRequest{node, 2, per_channel, x.dtype});
            requests.push_back(ScratchRequest{node, 3, per_channel, x.dtype});
            requests.push_back(ScratchRequest{node, 4, per_channel, x.dtype});
            requests.push_back(ScratchRequest{node, 5, x.shape, x.dtype});
            requests.push_back(ScratchRequest{node, 6, per_channel, x.dtype});
        }

        Status lower_batchnorm2d(const LoweringContext& context) noexcept
        {
            context.require_arity(3, 4);

            const Shape& x_shape = context.input_value(0).shape;
            SA_FATAL_IF(x_shape.rank() != 4,
                        "BATCHNORM2D (node {}) needs a rank-4 [N,H,W,C] operand; got {}",
                        context.id(), x_shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            SA_FATAL_IF(!x.is_contiguous(),
                        "BATCHNORM2D (node {}) folds its statistics over a rank-2 view of its "
                        "operand, which a strided view would read wrong", context.id());

            const SushiBLAS::Tensor& gamma = context.input(1);
            const SushiBLAS::Tensor& beta = context.input(2);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& mean, context.output(1));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& inv_std, context.output(2));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& variance, context.output(3));

            const DType dtype = context.input_value(0).dtype;
            const int64_t channels = x_shape[3];
            const int64_t rows = x_shape[0] * x_shape[1] * x_shape[2];
            const double inverse_rows = 1.0 / static_cast<double>(rows);
            const double epsilon = context.node().attributes.get(0);

            const Shape per_channel({channels});
            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones, context.ones(per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& negated_mean,
                          context.scratch(0, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& centred, context.scratch(1, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& scale, context.scratch(2, per_channel, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::SpatialOps spatial(context.recorder());

            const SushiBLAS::Tensor x_rows = channel_rows(x, rows, channels);
            reductions.sum_along(x_rows, 0, mean);
            blas.scal(inverse_rows, mean);

            blas.copy(mean, negated_mean);
            blas.scal(-1.0, negated_mean);
            spatial.channel_affine(x, ones, negated_mean, centred);

            // y holds the squares until their sums are taken; its own values are written last.
            elementwise.mul(centred, centred, y);
            const SushiBLAS::Tensor squared_rows = channel_rows(y, rows, channels);
            reductions.sum_along(squared_rows, 0, variance);
            blas.scal(inverse_rows, variance);

            // variance stays biased and without epsilon; only its copy in inv_std receives it.
            blas.copy(variance, inv_std);
            blas.axpy(epsilon, ones, inv_std);
            elementwise.sqrt(inv_std);
            elementwise.reciprocal(inv_std);

            elementwise.mul(gamma, inv_std, scale);
            spatial.channel_affine(centred, scale, beta, y);

            context.count(14);
            return Status();
        }

        Status lower_batchnorm2d_backward(const LoweringContext& context) noexcept
        {
            context.require_arity(5, 3);

            const Shape& x_shape = context.input_value(1).shape;
            SA_FATAL_IF(x_shape.rank() != 4,
                        "BATCHNORM2D_BACKWARD (node {}) needs a rank-4 [N,H,W,C] input; got {}",
                        context.id(), x_shape.to_string());

            const SushiBLAS::Tensor& dy = context.input(0);
            const SushiBLAS::Tensor& x = context.input(1);
            const SushiBLAS::Tensor& gamma = context.input(2);
            const SushiBLAS::Tensor& mean = context.input(3);
            const SushiBLAS::Tensor& inv_std = context.input(4);

            SA_FATAL_IF(!dy.is_contiguous() || !x.is_contiguous(),
                        "BATCHNORM2D_BACKWARD (node {}) folds over rank-2 views of its two "
                        "rank-4 operands, which a strided view would read wrong", context.id());

            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dgamma, context.output(1));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dbeta, context.output(2));

            const DType dtype = context.input_value(1).dtype;
            const int64_t channels = x_shape[3];
            const int64_t rows = x_shape[0] * x_shape[1] * x_shape[2];
            const double inverse_rows = 1.0 / static_cast<double>(rows);

            const Shape per_channel({channels});
            SA_TRY_ASSIGN(SushiBLAS::Tensor& shift, context.scratch(0, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& x_hat, context.scratch(1, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& direct, context.scratch(2, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& hat_term, context.scratch(3, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& mean_term, context.scratch(4, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& correction, context.scratch(5, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& zero, context.scratch(6, per_channel, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::SpatialOps spatial(context.recorder());
            SushiBLAS::RandomOps random(context.recorder(), context.stream());

            elementwise.mul(mean, inv_std, shift);
            blas.scal(-1.0, shift);
            spatial.channel_affine(x, inv_std, shift, x_hat);

            const SushiBLAS::Tensor dy_rows = channel_rows(dy, rows, channels);
            reductions.sum_along(dy_rows, 0, dbeta);

            // dx holds dy * x_hat until its sums are taken; its own values are written after.
            elementwise.mul(dy, x_hat, dx);
            const SushiBLAS::Tensor product_rows = channel_rows(dx, rows, channels);
            reductions.sum_along(product_rows, 0, dgamma);

            elementwise.mul(gamma, inv_std, direct);

            elementwise.mul(direct, dgamma, hat_term);
            blas.scal(-inverse_rows, hat_term);

            elementwise.mul(direct, dbeta, mean_term);
            blas.scal(-inverse_rows, mean_term);

            random.constant(zero, 0.0);
            spatial.channel_affine(dy, direct, zero, dx);
            spatial.channel_affine(x_hat, hat_term, mean_term, correction);
            blas.axpy(1.0, correction, dx);

            context.count(15);
            return Status();
        }

        Status lower_channel_affine(const LoweringContext& context) noexcept
        {
            context.require_arity(3, 1);

            const SushiBLAS::Tensor& x = context.input(0);
            const SushiBLAS::Tensor& gamma = context.input(1);
            const SushiBLAS::Tensor& beta = context.input(2);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

            SushiBLAS::SpatialOps spatial(context.recorder());
            spatial.channel_affine(x, gamma, beta, y);

            context.count(1);
            return Status();
        }

        Status lower_batchnorm_fold(const LoweringContext& context) noexcept
        {
            context.require_arity(4, 2);

            const SushiBLAS::Tensor& gamma = context.input(0);
            const SushiBLAS::Tensor& beta = context.input(1);
            const SushiBLAS::Tensor& mean = context.input(2);
            const SushiBLAS::Tensor& variance = context.input(3);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& scale, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& shift, context.output(1));

            const Shape& per_channel = context.input_value(3).shape;
            const DType dtype = context.input_value(3).dtype;
            const double epsilon = context.node().attributes.get(0);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones, context.ones(per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& inv_std, context.scratch(0, per_channel, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& product, context.scratch(1, per_channel, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());

            blas.copy(variance, inv_std);
            blas.axpy(epsilon, ones, inv_std);
            elementwise.sqrt(inv_std);
            elementwise.reciprocal(inv_std);

            elementwise.mul(gamma, inv_std, scale);
            elementwise.mul(mean, scale, product);
            elementwise.sub(beta, product, shift);

            context.count(7);
            return Status();
        }

        Status lower_layer_norm(const LoweringContext& context) noexcept
        {
            context.require_arity(3, 3);

            const RowExtents extents = row_extents(context, context.input_value(0).shape);
            const int64_t rows = extents.rows;
            const int64_t width = extents.width;

            const SushiBLAS::Tensor& x = context.input(0);
            SA_FATAL_IF(!x.is_contiguous(),
                        "LAYER_NORM (node {}) reads its operand as rows, which a strided view "
                        "would read wrong", context.id());

            const SushiBLAS::Tensor& gamma = context.input(1);
            const SushiBLAS::Tensor& beta = context.input(2);

            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& mean, context.output(1));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& inv_std, context.output(2));

            const DType dtype = context.input_value(0).dtype;
            const double inverse_width = 1.0 / static_cast<double>(width);
            const double epsilon = context.node().attributes.get(0);

            const Shape per_row({rows});
            const Shape matrix({rows, width});
            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones, context.ones(per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& negated_mean, context.scratch(0, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& zero, context.scratch(1, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& centred, context.scratch(2, matrix, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& squared, context.scratch(3, matrix, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::SpatialOps spatial(context.recorder());
            SushiBLAS::RandomOps random(context.recorder(), context.stream());

            const SushiBLAS::Tensor x_rows = channel_rows(x, rows, width);
            reductions.sum_along(x_rows, 1, mean);
            blas.scal(inverse_width, mean);

            blas.copy(mean, negated_mean);
            blas.scal(-1.0, negated_mean);
            SushiBLAS::Tensor centred_by_row = row_channels(centred, rows, width);
            spatial.channel_affine(row_channels(x, rows, width), ones, negated_mean,
                                   centred_by_row);

            elementwise.mul(centred, centred, squared);
            reductions.sum_along(squared, 1, inv_std);
            blas.scal(inverse_width, inv_std);
            blas.axpy(epsilon, ones, inv_std);
            elementwise.sqrt(inv_std);
            elementwise.reciprocal(inv_std);

            // squared is dead once its row sums are taken, so it holds the normalised rows.
            random.constant(zero, 0.0);
            SushiBLAS::Tensor normalised_by_row = row_channels(squared, rows, width);
            spatial.channel_affine(centred_by_row, inv_std, zero, normalised_by_row);

            SushiBLAS::Tensor y_columns = column_channels(y, rows, width);
            spatial.channel_affine(column_channels(squared, rows, width), gamma, beta, y_columns);

            context.count(14);
            return Status();
        }

        Status lower_layer_norm_backward(const LoweringContext& context) noexcept
        {
            context.require_arity(5, 3);

            const RowExtents extents = row_extents(context, context.input_value(1).shape);
            const int64_t rows = extents.rows;
            const int64_t width = extents.width;

            const SushiBLAS::Tensor& dy = context.input(0);
            const SushiBLAS::Tensor& x = context.input(1);
            const SushiBLAS::Tensor& gamma = context.input(2);
            const SushiBLAS::Tensor& mean = context.input(3);
            const SushiBLAS::Tensor& inv_std = context.input(4);

            SA_FATAL_IF(!dy.is_contiguous() || !x.is_contiguous(),
                        "LAYER_NORM_BACKWARD (node {}) reads its two [..., D] operands as rows, "
                        "which a strided view would read wrong", context.id());

            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dgamma, context.output(1));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dbeta, context.output(2));

            const DType dtype = context.input_value(1).dtype;
            const double inverse_width = 1.0 / static_cast<double>(width);

            const Shape per_row({rows});
            const Shape per_column({width});
            const Shape matrix({rows, width});
            SA_TRY_ASSIGN(SushiBLAS::Tensor& shift, context.scratch(0, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& x_hat, context.scratch(1, matrix, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& product, context.scratch(2, matrix, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& scaled, context.scratch(3, matrix, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_sum, context.scratch(4, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_dot, context.scratch(5, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& mean_term, context.scratch(6, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& hat_term, context.scratch(7, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& zero_row, context.scratch(8, per_row, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& zero_column,
                          context.scratch(9, per_column, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::SpatialOps spatial(context.recorder());
            SushiBLAS::RandomOps random(context.recorder(), context.stream());

            elementwise.mul(mean, inv_std, shift);
            blas.scal(-1.0, shift);
            SushiBLAS::Tensor x_hat_by_row = row_channels(x_hat, rows, width);
            spatial.channel_affine(row_channels(x, rows, width), inv_std, shift, x_hat_by_row);

            const SushiBLAS::Tensor dy_rows = channel_rows(dy, rows, width);
            reductions.sum_along(dy_rows, 0, dbeta);

            elementwise.mul(dy_rows, x_hat, product);
            reductions.sum_along(product, 0, dgamma);

            random.constant(zero_column, 0.0);
            random.constant(zero_row, 0.0);
            SushiBLAS::Tensor scaled_columns = column_channels(scaled, rows, width);
            spatial.channel_affine(column_channels(dy, rows, width), gamma, zero_column,
                                   scaled_columns);

            reductions.sum_along(scaled, 1, row_sum);
            elementwise.mul(scaled, x_hat, product);
            reductions.sum_along(product, 1, row_dot);

            elementwise.mul(row_sum, inv_std, mean_term);
            blas.scal(-inverse_width, mean_term);
            elementwise.mul(row_dot, inv_std, hat_term);
            blas.scal(-inverse_width, hat_term);

            // dx = inv_std * (g - mean(g) - x_hat * mean(g * x_hat)), with g = dy * gamma.
            SushiBLAS::Tensor dx_by_row = row_channels(dx, rows, width);
            spatial.channel_affine(row_channels(scaled, rows, width), inv_std, mean_term,
                                   dx_by_row);
            SushiBLAS::Tensor correction_by_row = row_channels(product, rows, width);
            spatial.channel_affine(x_hat_by_row, hat_term, zero_row, correction_by_row);

            SushiBLAS::Tensor dx_rows = channel_rows(dx, rows, width);
            blas.axpy(1.0, product, dx_rows);

            context.count(19);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
