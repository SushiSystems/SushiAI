/**************************************************************************/
/* conv_norm.hpp                                                          */
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
 * @file conv_norm.hpp
 * @brief Declares ConvNorm, a convolution followed by a batch normalisation and an activation.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /**
         * @brief Applies a bias-free convolution, a batch normalisation and an activation.
         * @tparam Convolution A layer with a static same_padded(in, out, kernel, stride).
         * @tparam Activation A default-constructible layer that holds no state.
         */
        template<typename Convolution, typename Activation>
        struct ConvNorm
        {
            /** @brief Holds the convolution, padded by kernel / 2 on every side. */
            Convolution conv;

            /** @brief Holds the normalisation over the convolution's output channels. */
            BatchNorm2d bn;

            /** @brief Holds the activation applied last. */
            Activation act{};

            /**
             * @brief Builds the three layers over the widths and the window given.
             * @param kernel The square window's side; an odd side keeps H and W at stride 1.
             * @param norm The epsilon and the momentum of the normalisation.
             */
            ConvNorm(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride,
                     NormOptions norm)
                : conv(Convolution::same_padded(in_channels, out_channels, kernel, stride)),
                  bn(out_channels, norm)
            {

            }

            /** @brief Lists the two children that hold state, under "conv" and "bn". */
            static constexpr auto fields()
            {
                return std::tuple{
                    child("conv", &ConvNorm::conv),
                    child("bn", &ConvNorm::bn),
                };
            }

            /**
             * @brief Traces the convolution, the normalisation in the trace's mode, then the
             *        activation.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The activated [N, out_h, out_w, out_channels] value's id.
             */
            Graph::ValueId forward(const Trace& trace, Graph::ValueId x)
            {
                x = trace_layer(conv, trace, x);
                x = trace_layer(bn, trace, x);
                return trace_layer(act, trace, x);
            }

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape that the window fits.
             */
            Shape output_shape(const Shape& input) const
            {
                return conv.output_shape(input);
            }
        };

        /** @brief Names the Ultralytics Conv: convolution, batch normalisation, SiLU. */
        using Conv = ConvNorm<Conv2d, SiLU>;

        /** @brief Names the Ultralytics Conv built with act=False, which ends at the norm. */
        using ConvLinear = ConvNorm<Conv2d, Identity>;

        static_assert(TraceModule<Conv>, "Conv must satisfy the TraceModule contract");
        static_assert(TraceModule<ConvLinear>, "ConvLinear must satisfy the TraceModule contract");

        /**
         * @brief Builds a Conv node from "filters", with optional "kernel" and "stride", 1 when
         *        absent, and the optional "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_conv_node(ShapeSpan inputs,
                                                                   const Config::Value& attributes,
                                                                   std::string_view context);

    } // namespace NN
} // namespace SushiAI
