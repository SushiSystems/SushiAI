/**************************************************************************/
/* matmul.cpp                                                             */
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
 * @file matmul.cpp
 * @brief Implements the lowering rows of the matrix product family: MATMUL, BIAS_ADD and SUM_ROWS.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/matmul.hpp>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/broadcast/broadcast.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status lower_matmul(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const SushiBLAS::Tensor& a = context.input(0);
            const SushiBLAS::Tensor& b = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& c, context.output(0));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.gemm(a, b, c, transpose_of(context.node().attributes.flag(0)),
                      transpose_of(context.node().attributes.flag(1)), 1.0, 0.0);
            context.count(1);
            return Status();
        }

        Status lower_bias_add(const LoweringContext& context) noexcept
        {
            context.require_arity(2, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "BIAS_ADD (node {}) needs a rank-2 left operand; got {}",
                        context.id(), x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            const SushiBLAS::Tensor& bias = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));

            // The bias meets column n of every row.
            SushiBLAS::BroadcastOps broadcast(context.recorder());
            broadcast.add(x, bias, y, SushiBLAS::BroadcastAxis::Trailing);
            context.count(1);
            return Status();
        }

        Status lower_sum_rows(const LoweringContext& context) noexcept
        {
            context.require_arity(1, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "SUM_ROWS needs a rank-2 operand; got {}", x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& sums, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones_rows,
                          context.ones(Shape{x_value.shape[0]}, x_value.dtype));

            SushiBLAS::BLASOps blas(context.recorder());
            blas.gemv(x, ones_rows, sums, SushiBLAS::Core::Transpose::Trans, 1.0f, 0.0f);
            context.count(1);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
