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
 * @file random.cpp
 * @brief Implements the lowering row of dropout and the positioned draw that redraws its mask.
 * @author Mustafa Garip
 */

#include <SushiAI/graph/lowering/random.hpp>

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/engine/rng_stream.hpp>
#include <SushiBLAS/random/positioned_random.hpp>
#include <SushiBLAS/random/rng_position_tensor.hpp>
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
            /** @brief Names the scratch slot of the INT64 position the draw reads. */
            constexpr int K_POSITION_SLOT = 0;

            /** @brief Names the scratch slot of the FLOAT32 draw a HALF mask is converted from. */
            constexpr int K_DRAW_SLOT = 1;

            /** @brief Names the result of a DROPOUT node that holds the mask it saves. */
            constexpr std::size_t K_MASK_RESULT = 1;

            /**
             * @brief Registers the prelude that writes the stream position of each replay.
             * @param position The INT64 tensor of K_RNG_POSITION_ELEMENTS elements the draw reads.
             * @param site The site seed of the node.
             * @return Success, or the failure of the registration.
             */
            [[nodiscard]] Status register_position_prelude(const LoweringContext& context,
                                                           SushiBLAS::Tensor& position,
                                                           std::uint64_t site)
            {
                SushiBLAS::Tensor* const target = &position;
                const SushiBLAS::RNGStream* const stream = &context.stream();

                // The seed word mixes the engine's seed with the site; the offset is the run.
                return context.before_each_replay([target, stream, site](std::uint64_t run)
                {
                    SushiBLAS::write_rng_position(
                        *target, SushiBLAS::RNGPosition{splitmix64_at(stream->seed(), site), run});
                });
            }

            /**
             * @brief Records the Bernoulli draw that fills @p mask with ones and zeros.
             * @param mask The tensor of result K_MASK_RESULT, which the draw lands in.
             * @param keep The probability of a one.
             * @param position The position tensor the prelude writes before each replay.
             * @return The number of tasks recorded, or the failure of an allocation.
             */
            [[nodiscard]] Result<std::size_t> record_keep_draw(const LoweringContext& context,
                                                               SushiBLAS::Tensor& mask,
                                                               double keep,
                                                               const SushiBLAS::Tensor& position)
            {
                SushiBLAS::PositionedRandomOps random(context.recorder());
                if (context.output_value(K_MASK_RESULT).dtype != DType::Half)
                {
                    random.bernoulli(mask, keep, position);
                    return std::size_t{1};
                }

                // The draw has no HALF form, so it lands in FLOAT32 and is converted.
                SA_TRY_ASSIGN(SushiBLAS::Tensor& draws,
                              context.scratch(K_DRAW_SLOT, context.output_value(K_MASK_RESULT).shape,
                                              DType::Float32));
                random.bernoulli(draws, keep, position);

                SushiBLAS::ElementwiseOps elementwise(context.recorder());
                elementwise.convert(draws, mask);
                return std::size_t{2};
            }
        } // namespace

        Status lower_dropout(const LoweringContext& context) noexcept
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
            SA_TRY_ASSIGN(SushiBLAS::Tensor& position,
                          context.scratch(K_POSITION_SLOT,
                                          Shape{SushiBLAS::K_RNG_POSITION_ELEMENTS},
                                          DType::Int64, TensorAccess::Host));

            SA_TRY(register_position_prelude(context, position,
                                             dropout_seed(context.node().attributes)));

            const double keep = 1.0 - rate;
            SA_TRY_ASSIGN(const std::size_t drawn, record_keep_draw(context, mask, keep, position));

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
