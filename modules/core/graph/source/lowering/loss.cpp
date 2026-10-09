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
 * @file loss.cpp
 * @brief Implements the lowering rows of the loss domain: LOG_SOFTMAX and NLL_LOSS.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/loss.hpp>

#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/reductions/reductions.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        Status lower_log_softmax(const LoweringContext& context)
        {
            context.require_arity(1, 1);

            const Value& description = context.input_value(0);
            const Shape& x_shape = description.shape;
            SA_FATAL_IF(x_shape.rank() != 2,
                        "LOG_SOFTMAX (node {}) needs a rank-2 operand; got {}",
                        context.id(), x_shape.to_string());

            const int64_t rows = x_shape[0];
            const int64_t cols = x_shape[1];
            const DType dtype = description.dtype;

            const SushiBLAS::Tensor& x = context.input(0);
            SA_TRY_ASSIGN(SushiBLAS::Tensor& out, context.output(0));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_max, context.scratch(0, Shape{rows}, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& shifted, context.scratch(1, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& exponents, context.scratch(2, x_shape, dtype));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& row_sums, context.scratch(3, Shape{rows}, dtype));

            SA_TRY_ASSIGN(SushiBLAS::Tensor& ones_cols, context.ones(Shape{cols}, dtype));

            SushiBLAS::ReductionOps reductions(context.recorder());
            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());

            reductions.max_along(x, 1, row_max);

            blas.copy(x, shifted);
            blas.ger(row_max, ones_cols, shifted, -1.0f);

            blas.copy(shifted, exponents);
            elementwise.exp(exponents);
            blas.gemv(exponents, ones_cols, row_sums, SushiBLAS::Core::Transpose::NoTrans, 1.0f,
                      0.0f);

            elementwise.log(row_sums);
            blas.copy(shifted, out);
            blas.ger(row_sums, ones_cols, out, -1.0f);

            context.count(9);
            return Status();
        }

        Status lower_nll_loss(const LoweringContext& context)
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
            blas.scal(-1.0f / static_cast<float>(description.shape[0]), loss);

            context.count(3);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
