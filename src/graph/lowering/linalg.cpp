/**************************************************************************/
/* linalg.cpp                                                             */
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
 * @file linalg.cpp
 * @brief Implements the lowering rows of the linear-algebra operations.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/linalg.hpp>

#include <SushiBLAS/engine/blas.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status lower_matmul(const LoweringContext& context)
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

        Status lower_bias_add(const LoweringContext& context)
        {
            context.require_arity(2, 1);

            const Value& x_value = context.input_value(0);
            SA_FATAL_IF(x_value.shape.rank() != 2,
                        "BIAS_ADD (node {}) needs a rank-2 left operand; got {}",
                        context.id(), x_value.shape.to_string());

            const SushiBLAS::Tensor& x = context.input(0);
            const SushiBLAS::Tensor& bias = context.input(1);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& rows,
                          context.ones(Shape{x_value.shape[0]}, x_value.dtype));

            SushiBLAS::BLASOps blas(context.recorder());

            // ger adds the outer product to y, so y holds x before it runs.
            blas.copy(x, y);
            blas.ger(rows, bias, y, 1.0f);
            context.count(2);
            return Status();
        }

        Status lower_sum_rows(const LoweringContext& context)
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
