/**************************************************************************/
/* channel_slice.hpp                                                      */
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
 * @file channel_slice.hpp
 * @brief Declares the layer that keeps a contiguous range of channels of an NHWC image.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/module.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Keeps the channels [offset, offset + count) of an NHWC image. */
        struct ChannelSlice
        {
            /**
             * @brief Holds the index of the first channel kept; must be at least 0 and below the
             * input's channel count.
             */
            int64_t offset = 0;

            /**
             * @brief Holds the number of channels kept; must be at least 1 and must not run past
             * the input's channel count.
             */
            int64_t count = 0;

            /** @brief Lists this layer's parameters, of which there are none. */
            static constexpr auto fields() noexcept
            {
                return std::tuple<>{};
            }

            /**
             * @brief Traces the channel slice of @p input.
             * @param input An NHWC [N, H, W, C] value.
             * @return The [N, H, W, count] result; the failure of @p input,
             *         or InvalidDescription for an input the layer refuses.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input) const noexcept;

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, C] shape.
             * @return The per-sample [H, W, count] shape, or InvalidDescription for a
             *         slice that does not fit @p input.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(Module<ChannelSlice>, "ChannelSlice must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
