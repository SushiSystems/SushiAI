/**************************************************************************/
/* conv2d.hpp                                                             */
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
 * @file conv2d.hpp
 * @brief Declares the 2-D convolution layer that traces into graph IR.
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
        /** @brief Convolves an NHWC activation with a bank of learnable filters. */
        struct Conv2d
        {
            /** @brief Holds the channel count of the input. */
            int64_t in_channels = 0;

            /** @brief Holds the number of filters, which is the channel count of the result. */
            int64_t out_channels = 0;

            /** @brief Holds the window; its top and left pads apply to both sides of an axis. */
            Graph::ConvGeometry geometry{};

            /** @brief Tells whether the layer adds a learnable bias per filter. */
            bool use_bias = true;

            /**
             * @brief Holds the value every bias element starts from; forward() reads it on the
             *        first trace only, and never without a bias.
             */
            double bias_init = 0.0;

            /** @brief Holds the [out_channels, kernel_h, kernel_w, in_channels] filters. */
            Parameter weight{};

            /** @brief Holds the [out_channels] bias; undeclared when use_bias is false. */
            Parameter bias{};

            /** @brief Lists this layer's parameters for the generic walk. */
            static constexpr auto fields()
            {
                return std::tuple{
                    field("weight", &Conv2d::weight),
                    field("bias", &Conv2d::bias),
                };
            }

            /**
             * @brief Builds a bias-free convolution padded by kernel / 2 on every side.
             * @param kernel The square window's side; an odd side keeps H and W at stride 1.
             * @param stride The step on both axes; 2 halves an even extent under an odd kernel.
             * @return A layer with dilation 1 whose geometry output_shape() and forward() check.
             */
            [[nodiscard]] static Conv2d same_padded(int64_t in_channels, int64_t out_channels,
                                                    int64_t kernel, int64_t stride);

            /**
             * @brief Traces the convolution, declaring its parameters on the first call.
             * @param input An NHWC [N, H, W, in_channels] value that the window fits.
             * @return The [N, out_h, out_w, out_channels] result; the failure of @p input,
             *         or InvalidDescription for an input the layer refuses.
             */
            Graph::Traced forward(const Graph::GraphBuilder& builder,
                                  const Graph::Traced& input);

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape that the window fits.
             * @return The per-sample [out_h, out_w, out_channels] shape, or
             *         InvalidDescription for a geometry or a shape the window does not fit.
             */
            Result<Shape> output_shape(const Shape& input) const;
        };

        static_assert(Module<Conv2d>, "Conv2d must satisfy the Module contract");

    } // namespace NN
} // namespace SushiAI
