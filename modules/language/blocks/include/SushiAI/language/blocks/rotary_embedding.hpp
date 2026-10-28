/**************************************************************************/
/* rotary_embedding.hpp                                                   */
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
 * @file rotary_embedding.hpp
 * @brief Declares RotaryEmbedding, which turns the channel pairs of a sequence by position.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /**
         * @brief Turns each channel pair of a [N, T, width] value by the angle of its position.
         *
         * The width is cut into groups of equal size. Pair i of a group is channel i of its
         * first half with channel i of its second half, and turns by rotary_angle().
         */
        struct RotaryEmbedding
        {
            /** @brief Holds the channels of the input, which are those of the result. */
            int64_t width = 0;

            /** @brief Holds the parts of the width that turn alike. */
            int64_t groups = 0;

            /** @brief Builds a block of no width, which no input fits. */
            RotaryEmbedding() = default;

            /** @brief Describes the block; forward() checks the widths. */
            RotaryEmbedding(int64_t width, int64_t groups) noexcept;

            /** @brief Lists no field: the block holds no state. */
            static constexpr auto fields()
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the turn of row t by position t, with the cosines and the sines
             *        of every row in two host tables of the trace.
             * @param x A [N, T, width] value of a real floating type.
             * @return The [N, T, width] result's id, or @p x after trace.fail() when @p x,
             *         this block's description or the trace does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces the turn of one position, whose cosines and sines the run feeds
             *        under K_ROTARY_COSINE_LEAF and K_ROTARY_SINE_LEAF.
             * @param x The [1, 1, width] slice of one sequence, of a real floating type.
             * @param sample The per-sample [T, width] shape of the ordinary form.
             * @return The [1, 1, width] result's id, or @p x after trace.fail().
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Result<Shape> output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<RotaryEmbedding>,
                      "RotaryEmbedding must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<RotaryEmbedding>,
                      "RotaryEmbedding must have an incremental form");

        /**
         * @brief Returns InvalidDescription unless @p groups is positive and cuts @p width
         *        into parts of an even, positive number of channels.
         */
        [[nodiscard]] Status check_rotary_widths(int64_t width, int64_t groups);

    } // namespace Language::Blocks
} // namespace SushiAI
