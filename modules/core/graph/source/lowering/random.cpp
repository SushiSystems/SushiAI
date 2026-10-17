/**************************************************************************/
/* random.cpp                                                             */
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
 * @file random.cpp
 * @brief Implements the lowering row of dropout and the fill that redraws its mask.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/random.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering/context.hpp>

namespace SushiAI
{
    namespace Graph
    {
        namespace
        {
            /** @brief Names the scratch slot of the FLOAT32 draws the host writes. */
            constexpr int K_HOST_DRAW_SLOT = 0;

            /** @brief Names the result of a DROPOUT node that holds the mask it saves. */
            constexpr std::size_t K_MASK_RESULT = 1;

            /**
             * @brief Records what makes @p mask hold a new draw of ones and zeros at each replay.
             * @param mask The tensor of result K_MASK_RESULT, which the draw lands in.
             * @param keep The probability of a one.
             * @param seed The site seed the draws are a function of, with the run and the index.
             * @return The number of tasks recorded, or the failure of an allocation.
             */
            [[nodiscard]] Result<std::size_t> record_keep_draw(const LoweringContext& context,
                                                               SushiBLAS::Tensor& mask,
                                                               double keep, std::uint64_t seed)
            {
                const Value& described = context.output_value(K_MASK_RESULT);
                SA_TRY_ASSIGN(SushiBLAS::Tensor& draws,
                              context.scratch(K_HOST_DRAW_SLOT, described.shape,
                                              DType::Float32));

                // No task writes draws, so the host does, while no replay is in flight.
                SushiBLAS::Tensor* const target = &draws;
                const std::size_t count = static_cast<std::size_t>(draws.num_elements);
                SA_TRY(context.before_each_replay([target, count, seed, keep](std::uint64_t run)
                {
                    fill_keep_mask(target->data_as<float>(), count, seed, run, keep);
                }));

                if (described.dtype == DType::Float32)
                {
                    SushiBLAS::BLASOps blas(context.recorder());
                    blas.copy(draws, mask);
                }
                else
                {
                    SushiBLAS::ElementwiseOps elementwise(context.recorder());
                    elementwise.convert(draws, mask);
                }

                return std::size_t{1};
            }
        } // namespace

        Status lower_dropout(const LoweringContext& context)
        {
            context.require_arity(1, 2);

            const double rate = context.node().attributes.get(K_DROPOUT_RATE_SLOT);
            SA_FATAL_IF(!(rate > 0.0 && rate < 1.0),
                        "DROPOUT (node {}) needs a rate in (0, 1); got {}", context.id(), rate);

            const SushiBLAS::Tensor& x = context.input(0);
            SA_FATAL_IF(!x.is_contiguous(),
                        "DROPOUT (node {}) reads its operand element by element, which a "
                        "strided view would pair with the wrong draws", context.id());

            SA_TRY_ASSIGN(SushiBLAS::Tensor& y, context.output(0));
            SA_TRY_ASSIGN(SushiBLAS::Tensor& mask, context.output(K_MASK_RESULT));

            const double keep = 1.0 - rate;
            SA_TRY_ASSIGN(const std::size_t drawn,
                          record_keep_draw(context, mask, keep,
                                           dropout_seed(context.node().attributes)));

            SushiBLAS::BLASOps blas(context.recorder());
            SushiBLAS::ElementwiseOps elementwise(context.recorder());

            // A kept element is scaled by 1 / keep, so the expectation of y is x.
            blas.scal(1.0 / keep, mask);
            elementwise.mul(x, mask, y);

            context.count(drawn + 2);
            return Status();
        }
    } // namespace Graph
} // namespace SushiAI
