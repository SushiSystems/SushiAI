/**************************************************************************/
/* sppf.cpp                                                               */
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
 * @file sppf.cpp
 * @brief Implements SPPF's trace and the factory that builds its node.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/blocks/sppf.hpp>

#include <cstddef>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Vision::Blocks
    {
        namespace
        {
            /** @brief Lists every key an "sppf" row accepts. */
            constexpr std::string_view K_SPPF_KEYS[] = {
                "filters", "kernel", "pools", "shortcut", Detail::K_NORM_EPSILON_KEY,
                Detail::K_NORM_MOMENTUM_KEY};

            /** @brief Holds the pool's window side in a row that states none. */
            constexpr int64_t K_DEFAULT_KERNEL = 5;

            /** @brief Holds the pool count of a row that states none. */
            constexpr int64_t K_DEFAULT_POOLS = 3;

            /** @brief Holds whether an SPPF adds its input in a row that does not say. */
            constexpr bool K_SPPF_DEFAULT_SHORTCUT = false;

            /** @brief Holds the smallest input width, which leaves a hidden width of one. */
            constexpr int64_t K_SMALLEST_INPUT_WIDTH = 2;

            /** @brief Returns the stride-1 window of side @p kernel padded by kernel / 2. */
            [[nodiscard]] Graph::ConvGeometry pool_geometry(int64_t kernel) noexcept
            {
                Graph::ConvGeometry geometry;
                geometry.kernel_h = kernel;
                geometry.kernel_w = kernel;
                geometry.pad_top = kernel / 2;
                geometry.pad_left = kernel / 2;
                return geometry;
            }

        } // namespace

        SPPF::SPPF(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t pool_count,
                   bool shortcut, NN::NormOptions norm)
            : cv1(in_channels, in_channels / 2, Detail::K_POINTWISE, Detail::K_POINTWISE, norm),
              cv2((in_channels / 2) * (pool_count + 1), out_channels, Detail::K_POINTWISE,
                  Detail::K_POINTWISE, norm),
              pool{pool_geometry(kernel)},
              pools(pool_count),
              adds_input(shortcut && in_channels == out_channels)
        {

        }

        Graph::ValueId SPPF::forward(const NN::Trace& trace, Graph::ValueId x)
        {
            const Graph::GraphBuilder& builder = trace.builder();

            std::vector<Graph::ValueId> parts;
            parts.reserve(static_cast<std::size_t>(pools) + 1);
            parts.push_back(NN::trace_layer(cv1, trace, x));
            for (int64_t index = 0; index < pools; ++index)
                parts.push_back(NN::trace_layer(pool, trace, parts.back()));

            const Graph::ValueId y = NN::trace_layer(
                cv2, trace,
                builder.concat_channels(
                    SushiRuntime::span<const Graph::ValueId>(parts.data(), parts.size())));
            if (!adds_input)
                return y;

            return builder.add(y, x);
        }

        Shape SPPF::output_shape(const Shape& input) const
        {
            // cv1 refuses an input that is not [H, W, in_channels] and keeps H and W.
            const Shape narrowed = cv1.output_shape(input);

            return cv2.output_shape(Shape{narrowed[0], narrowed[1], cv2.conv.in_channels});
        }

        Result<std::unique_ptr<NN::Node>> make_sppf_node(NN::ShapeSpan inputs,
                                                         const Config::Value& attributes,
                                                         std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_SPPF_KEYS), context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_feature_map(inputs, context));
            SA_TRY_ASSIGN(const int64_t filters,
                          Detail::read_positive_int(attributes, "filters", context));
            SA_TRY_ASSIGN(const int64_t kernel,
                          Detail::read_positive_int_or(attributes, "kernel", context,
                                                       K_DEFAULT_KERNEL));
            SA_TRY_ASSIGN(const int64_t pool_count,
                          Detail::read_positive_int_or(attributes, "pools", context,
                                                       K_DEFAULT_POOLS));
            SA_TRY_ASSIGN(const bool shortcut,
                          Detail::read_bool_or(attributes, "shortcut", context,
                                               K_SPPF_DEFAULT_SHORTCUT));
            SA_TRY_ASSIGN(const NN::NormOptions norm,
                          Detail::read_norm_options(attributes, context));

            SA_FAIL_IF(kernel % 2 == 0, StatusCode::InvalidDescription,
                       "{} 'kernel' must be odd, so that the pool keeps H and W; it is {}",
                       context, kernel);
            SA_FAIL_IF(input[2] < K_SMALLEST_INPUT_WIDTH, StatusCode::InvalidDescription,
                       "{} was given a per-sample input of {}, but it needs at least {} "
                       "channels, because the hidden width is half of them",
                       context, input.to_string(), K_SMALLEST_INPUT_WIDTH);

            return Detail::wrap_block(SPPF(input[2], filters, kernel, pool_count, shortcut, norm),
                                      input);
        }

    } // namespace Vision::Blocks
} // namespace SushiAI
