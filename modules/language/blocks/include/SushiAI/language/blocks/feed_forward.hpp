/**************************************************************************/
/* feed_forward.hpp                                                       */
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
 * @file feed_forward.hpp
 * @brief Declares FeedForward, a linear layer, a GELU and a linear layer back to the width.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Maps each position of a [N, T, dim] sequence through two linear layers. */
        struct FeedForward
        {
            /** @brief Holds the width of the input, which is that of the result. */
            int64_t dim = 0;

            /** @brief Holds the width between the two linear layers. */
            int64_t hidden = 0;

            /** @brief Holds the linear layer from dim to hidden. */
            NN::Linear expand{};

            /** @brief Holds the activation between the two linear layers. */
            NN::GELU activation{};

            /** @brief Holds the linear layer from hidden back to dim. */
            NN::Linear project{};

            /** @brief Builds a block of no width, which no input fits. */
            FeedForward() = default;

            /** @brief Builds the two linear layers; forward() checks the widths. */
            FeedForward(int64_t dim, int64_t hidden) noexcept;

            /** @brief Lists the two children that hold state, under "expand" and "project". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("expand", &FeedForward::expand),
                    NN::child("project", &FeedForward::project),
                };
            }

            /**
             * @brief Traces expand, the GELU and project over the [N * T, dim] rows of @p x.
             * @param x A [N, T, dim] value of a real floating type.
             * @return The [N, T, dim] result's id, or @p x after trace.fail() when @p x or
             *         this block's description does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces forward() over the newest slice, as a position reads no other.
             * @param sample Not read.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /** @brief Returns the per-sample shape forward() produces, which is @p input. */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<FeedForward>,
                      "FeedForward must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<FeedForward>,
                      "FeedForward must have an incremental form");

        /**
         * @brief Builds a FeedForward node from the required key "hidden".
         * @param inputs One per-sample [T, C] shape; C becomes dim.
         * @return The node, or InvalidDescription when @p inputs or the key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_feed_forward_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Language::Blocks
} // namespace SushiAI
