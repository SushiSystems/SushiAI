/**************************************************************************/
/* loss.cpp                                                               */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file loss.cpp
 * @brief Implements the lowering rows of the loss domain: LOG_SOFTMAX, its gradient and NLL_LOSS.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/loss.hpp>

#include <array>
#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/broadcast/broadcast.hpp>
#include <SushiBLAS/broadcast/dispatch.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/ops/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status lower_log_softmax(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const Value& description = context.input_value(0);
            const Shape& x_shape = description.shape;
            SA_FATAL_IF(x_shape.rank() != 2,
                        "LOG_SOFTMAX (node {}) needs a rank-2 operand; got {}",
                        context.id(), x_shape.to_string());

            const int64_t rows = x_shape[0];
            const DType dtype = description.dtype;

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_max, context.scratch(0, Shape{rows}, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& shifted, context.scratch(1, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& exponents, context.scratch(2, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_sums, context.scratch(3, Shape{rows}, dtype));

            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::BroadcastOps broadcast(context.recorder());

            reductions.max_along(x, 1, row_max);

            // Each subtraction takes one value per row from its own place, out of place.
            broadcast.sub(x, row_max, shifted, SushiBLAS::BroadcastAxis::Leading);

            blas.copy(shifted, exponents);
            elementwise.exp(exponents);
            reductions.sum_along(exponents, 1, row_sums);

            elementwise.log(row_sums);
            broadcast.sub(shifted, row_sums, out, SushiBLAS::BroadcastAxis::Leading);

            context.count(7);
            return Status();
        }

        Status lower_log_softmax_backward(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const Value& description = context.input_value(1);
            const Shape& shape = description.shape;
            SA_FATAL_IF(shape.rank() != 2,
                        "LOG_SOFTMAX_BACKWARD (node {}) needs rank-2 operands; got {}",
                        context.id(), shape.to_string());

            const int64_t rows = shape[0];
            const DType dtype = description.dtype;

            const SushiBLAS::Tensor& dy = context.input(0);
            const SushiBLAS::Tensor& log_probabilities = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& dx, context.output(0));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_sums, context.scratch(0, Shape{rows}, dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());

            reductions.sum_along(dy, 1, row_sums);

            blas.copy(log_probabilities, dx);
            elementwise.exp(dx);

            // dx = dy - exp(log_probabilities) * rowsum(dy): the call leaves -exp(.) * rowsum(dy).
            constexpr const char* K_TASK_NAME = Ops::K_LOG_SOFTMAX_BACKWARD.name.data();
            const std::array<SushiBLAS::BroadcastOperand, 1> row_sum_vector{
                {{&row_sums, SushiBLAS::BroadcastAxis::Leading}}};
            SushiBLAS::Dispatch::execute_broadcast(
                context.recorder(), dx, row_sum_vector, dx, K_TASK_NAME,
                Ops::K_LOG_SOFTMAX_BACKWARD.id,
                [](auto probability, auto row_sum)
                {
                    return -probability * row_sum;
                });
            blas.axpy(1.0, dy, dx);

            context.count(5);
            return Status();
        }

        Status lower_nll_loss(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const Value& description = context.input_value(0);
            SA_FATAL_IF(description.shape.rank() != 2,
                        "NLL_LOSS (node {}) needs rank-2 operands; got {}",
                        context.id(), description.shape.to_string());

            const SushiBLAS::Tensor& log_probabilities = context.input(0);
            const SushiBLAS::Tensor& targets = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& loss, context.output(0));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& picked,
                          context.scratch(0, description.shape, description.dtype));

            SushiBLAS::ElementwiseOps elementwise(context.recorder());
            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::BLASOps blas(context.recorder());

            elementwise.mul(log_probabilities, targets, picked);
            reductions.sum(picked, loss);
            blas.scal(-1.0 / static_cast<double>(description.shape[0]), loss);

            context.count(3);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
