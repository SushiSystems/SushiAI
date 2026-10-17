/**************************************************************************/
/* batch_norm2d.hpp                                                       */
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
 * @file batch_norm2d.hpp
 * @brief Declares BatchNorm2d, its options, its node factory and the host form of an update.
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
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Holds the two constants of a batch normalisation. */
        struct NormOptions
        {
            /** @brief Holds the constant added to a variance before its square root. */
            double epsilon = 1e-5;

            /** @brief Weighs a batch statistic against the running statistic it moves. */
            double momentum = 0.1;
        };

        /** @brief Normalises an NHWC activation per channel. */
        struct BatchNorm2d
        {
            /** @brief Holds the channel count of the input, which is that of the result. */
            int64_t channels = 0;

            /** @brief Holds epsilon and momentum. */
            NormOptions options{};

            /** @brief Holds gamma, the [channels] SCALE parameter, which starts at one. */
            Parameter weight{};

            /** @brief Holds beta, the [channels] shift, which starts at zero. */
            Parameter bias{};

            /** @brief Holds the [channels] running mean, which starts at zero. */
            Buffer running_mean{};

            /** @brief Holds the [channels] running variance, which starts at one. */
            Buffer running_var{};

            /** @brief Builds a layer of no channels, which no input fits. */
            BatchNorm2d() = default;

            /** @brief Builds a layer over @p channels channels with the two constants given. */
            BatchNorm2d(int64_t channels, NormOptions options) noexcept;

            /** @brief Lists this layer's parameters and buffers for the state walk. */
            static constexpr auto fields()
            {
                return std::tuple{
                    field("weight", &BatchNorm2d::weight),
                    field("bias", &BatchNorm2d::bias),
                    buffer("running_mean", &BatchNorm2d::running_mean),
                    buffer("running_var", &BatchNorm2d::running_var),
                };
            }

            /**
             * @brief Traces BATCHNORM2D over batch statistics and requests both running updates.
             * @param trace A Training trace; an Evaluation trace is fatal.
             * @param x An NHWC [N, H, W, channels] value with N * H * W of at least two.
             * @return The normalised value's id, or @p x after trace.fail() when @p x or this
             *         layer's description does not fit.
             */
            Graph::ValueID trace_training(const Trace& trace, Graph::ValueID x);

            /**
             * @brief Traces BATCHNORM_FOLD over the running statistics, then CHANNEL_AFFINE.
             * @param x An NHWC [N, H, W, channels] value.
             * @return The normalised value's id, or @p x after trace.fail() when @p x or this
             *         layer's description does not fit.
             */
            Graph::ValueID trace_evaluation(const Trace& trace, Graph::ValueID x);

            /**
             * @brief Returns the per-sample shape either form produces, which is @p input.
             * @param input Any shape; a trace form refuses one that is not [H, W, channels].
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(ModalModule<BatchNorm2d>,
                      "BatchNorm2d must satisfy the ModalModule contract");

        /**
         * @brief Builds a BatchNorm2d node with the optional "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the channel count.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_batch_norm2d_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

        /**
         * @brief Returns (1 - momentum) * running + momentum * source_scale * batch.
         * @param source_scale The factor the batch statistic carries into the running one.
         */
        [[nodiscard]] double updated_statistic(double running, double batch, double momentum,
                                               double source_scale) noexcept;

    } // namespace NN
} // namespace SushiAI
