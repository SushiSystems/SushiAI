/**************************************************************************/
/* layer_norm.hpp                                                         */
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
 * @file layer_norm.hpp
 * @brief Declares LayerNorm, the layer that normalises each row of its input over the last axis.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Holds the epsilon of a layer normalisation that states none. */
        inline constexpr double K_LAYER_NORM_EPSILON = 1e-5;

        /** @brief Normalises each [dim] row of a [..., dim] value by its own mean and variance. */
        struct LayerNorm
        {
            /** @brief Holds the last extent of the input, which is that of the result. */
            int64_t dim = 0;

            /** @brief Holds the constant added to a row's variance before its square root. */
            double epsilon = K_LAYER_NORM_EPSILON;

            /** @brief Holds the gain, the [dim] SCALE parameter, which starts at one. */
            Parameter weight{};

            /** @brief Holds the [dim] bias, which starts at zero. */
            Parameter bias{};

            /** @brief Lists this layer's two parameters for the state walk. */
            static constexpr auto fields()
            {
                return std::tuple{
                    field("weight", &LayerNorm::weight),
                    field("bias", &LayerNorm::bias),
                };
            }

            /**
             * @brief Traces K_LAYER_NORM over the gain and the bias, declaring both on the
             *        first call.
             * @param x A [..., dim] value of a real floating type, batch axis included.
             * @return The normalised value's id, or @p x after trace.fail() when @p x or this
             *         layer's description does not fit.
             */
            Graph::ValueID forward(const Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces forward() over the newest slice, as a row reads no other row.
             * @param sample Not read.
             */
            Graph::ValueID forward_incremental(const Trace& trace, Graph::ValueID x,
                                               const Shape& sample);

            /**
             * @brief Returns the per-sample shape forward() produces, which is @p input.
             * @param input Any shape; forward() refuses one whose last extent is not dim.
             */
            Result<Shape> output_shape(const Shape& input) const;
        };

        static_assert(TraceModule<LayerNorm>, "LayerNorm must satisfy the TraceModule contract");
        static_assert(IncrementalModule<LayerNorm>, "LayerNorm must have an incremental form");

        /**
         * @brief Builds a LayerNorm node with the optional key "ln_epsilon".
         * @param inputs One per-sample shape of rank 1 to K_MAX_RANK - 1; its last extent
         *        becomes dim.
         * @param context What to call the row in a diagnostic, e.g. "layer 0 'layernorm'".
         * @return The node, or InvalidDescription when @p inputs or the key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_layer_norm_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace NN
} // namespace SushiAI
