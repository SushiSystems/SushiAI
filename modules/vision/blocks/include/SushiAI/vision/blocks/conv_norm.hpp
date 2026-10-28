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
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/depthwise_conv2d.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /**
         * @brief Applies a bias-free convolution, a batch normalisation and an activation.
         * @tparam Convolution A layer with a static same_padded(in, out, kernel, stride).
         * @tparam Activation A default-constructible layer that holds no state.
         */
        template<typename Convolution, typename Activation>
        struct ConvNorm
        {
            /** @brief Holds the convolution, padded by kernel / 2 on every side unless stated. */
            Convolution conv;

            /** @brief Holds the normalisation over the convolution's output channels. */
            NN::BatchNorm2d bn;

            /** @brief Holds the activation applied last. */
            Activation act{};

            /**
             * @brief Builds the three layers over the widths and the window given.
             * @param kernel The square window's side; an odd side keeps H and W at stride 1.
             * @param norm The epsilon and the momentum of the normalisation.
             */
            ConvNorm(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride,
                     NN::NormOptions norm) noexcept
                : conv(Convolution::same_padded(in_channels, out_channels, kernel, stride)),
                  bn(out_channels, norm)
            {

            }

            /**
             * @brief Builds the three layers over a window padded by @p padding on every side.
             * @param padding Rows and columns of zeros on each side; zero or more.
             */
            ConvNorm(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride,
                     int64_t padding, NN::NormOptions norm) noexcept
                : ConvNorm(in_channels, out_channels, kernel, stride, norm)
            {
                conv.geometry.pad_top = padding;
                conv.geometry.pad_left = padding;
            }

            /** @brief Lists the two children that hold state, under "conv" and "bn". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("conv", &ConvNorm::conv),
                    NN::child("bn", &ConvNorm::bn),
                };
            }

            /**
             * @brief Traces the convolution, the normalisation in the trace's mode, then the
             *        activation.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The activated [N, out_h, out_w, out_channels] value's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept
            {
                x = NN::trace_layer(conv, trace, x);
                x = NN::trace_layer(bn, trace, x);
                return NN::trace_layer(act, trace, x);
            }

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape that the window fits.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept
            {
                return conv.output_shape(input);
            }
        };

        /** @brief Names the Ultralytics Conv: convolution, batch normalisation, SiLU. */
        using Conv = ConvNorm<NN::Conv2d, NN::SiLU>;

        /** @brief Names the Ultralytics Conv built with act=False, which ends at the norm. */
        using ConvLinear = ConvNorm<NN::Conv2d, NN::Identity>;

        /** @brief Names the Ultralytics DWConv: depthwise convolution, batch norm, SiLU. */
        using DWConv = ConvNorm<NN::DepthwiseConv2d, NN::SiLU>;

        /** @brief Names the Ultralytics DWConv built with act=False, which ends at the norm. */
        using DWConvLinear = ConvNorm<NN::DepthwiseConv2d, NN::Identity>;

        static_assert(NN::TraceModule<Conv>, "Conv must satisfy the TraceModule contract");
        static_assert(NN::TraceModule<ConvLinear>,
                      "ConvLinear must satisfy the TraceModule contract");
        static_assert(NN::TraceModule<DWConv>, "DWConv must satisfy the TraceModule contract");
        static_assert(NN::TraceModule<DWConvLinear>,
                      "DWConvLinear must satisfy the TraceModule contract");

        /**
         * @brief Builds a Conv node from "filters", with optional "kernel" and "stride", 1 when
         *        absent, an optional "padding" of zero or more, kernel / 2 when absent, and the
         *        optional "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_conv_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context) noexcept;

    } // namespace Vision::Blocks
} // namespace SushiAI
