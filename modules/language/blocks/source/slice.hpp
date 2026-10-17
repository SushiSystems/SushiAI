/**************************************************************************/
/* slice.hpp                                                              */
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
 * @file slice.hpp
 * @brief Defines the checks and the position leaf the incremental forms of the blocks share.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Names the rank of the per-sample [T, C] shape a block reads. */
            inline constexpr int32_t K_WINDOW_RANK = 2;

            /**
             * @brief Checks that @p x is the [1, 1, dim] slice of one sequence.
             * @param x A value of builder.graph(); anything else is fatal.
             * @param block What to call the block in a diagnostic.
             * @return InvalidDescription when the extents or the element type do not fit.
             */
            [[nodiscard]] inline Status check_slice(const Graph::GraphBuilder& builder,
                                                    Graph::ValueID x, int64_t dim,
                                                    std::string_view block)
            {
                SA_FATAL_IF(x >= builder.graph().value_count(),
                            "{} was traced over value {}; the graph holds {} values", block, x,
                            builder.graph().value_count());

                const Shape shape = builder.graph().value(x).shape;
                const DType dtype = builder.graph().value(x).dtype;
                SA_FAIL_IF(shape != (Shape{1, 1, dim}), StatusCode::InvalidDescription,
                           "the incremental form of {} reads one position of one sequence, "
                           "[1, 1, {}]; got {}",
                           block, dim, shape.to_string());
                SA_FAIL_IF(!is_real_float(dtype), StatusCode::InvalidDescription,
                           "{} needs a half, float32 or float64 input; got {}", block,
                           dtype_name(dtype));

                return Status();
            }

            /**
             * @brief Returns T of the per-sample [T, dim] shape the block reads in its
             *        ordinary form.
             * @return InvalidDescription when @p sample is not [T, dim] with a positive T.
             */
            [[nodiscard]] inline Result<int64_t> window_of(const Shape& sample, int64_t dim,
                                                           std::string_view block)
            {
                SA_FAIL_IF(sample.rank() != K_WINDOW_RANK || sample[0] < 1 || sample[1] != dim,
                           StatusCode::InvalidDescription,
                           "the incremental form of {} needs the per-sample [T, {}] shape of "
                           "its ordinary form; got {}",
                           block, dim, sample.to_string());

                return sample[0];
            }

            /** @brief Returns the [1, 1] int32 leaf K_POSITION_LEAF, declared once per trace. */
            [[nodiscard]] inline Result<Graph::ValueID> position_leaf(const NN::Trace& trace)
            {
                return trace.feed(K_POSITION_LEAF, Shape{1, 1}, DType::Int32);
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
