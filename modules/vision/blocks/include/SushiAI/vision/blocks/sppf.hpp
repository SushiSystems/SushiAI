/**************************************************************************/
/* sppf.hpp                                                               */
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
 * @file sppf.hpp
 * @brief Declares SPPF, which pools a narrowed activation repeatedly and joins every stage.
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
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /** @brief Pools a narrowed activation repeatedly and joins every stage. */
        struct SPPF
        {
            /** @brief Holds the 1x1 convolution to half the input width, with no activation. */
            ConvLinear cv1;

            /** @brief Holds the 1x1 convolution over the narrowed value and every pooled one. */
            Conv cv2;

            /** @brief Holds the stride-1 pool, padded by kernel / 2 so that it keeps H and W. */
            NN::MaxPool2d pool;

            /** @brief Holds how many times the pool is applied, each time to its last result. */
            int64_t pools = 3;

            /** @brief Tells whether forward() adds its input to the second convolution's result. */
            bool adds_input = false;

            /**
             * @brief Builds both convolutions and the pool.
             * @param kernel The side of the pool's window; odd, so that H and W are kept.
             * @param pool_count How many times forward() pools.
             * @param shortcut Asks for the add, which is made only when in_channels equals
             *                 out_channels.
             */
            SPPF(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t pool_count,
                 bool shortcut, NN::NormOptions norm);

            /** @brief Lists the two children that hold state, under "cv1" and "cv2". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("cv1", &SPPF::cv1),
                    NN::child("cv2", &SPPF::cv2),
                };
            }

            /**
             * @brief Traces cv1, the pools in a chain, cv2 over all stages joined, then the add
             *        of @p x when adds_input holds.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H, W, out_channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x);

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(NN::TraceModule<SPPF>, "SPPF must satisfy the TraceModule contract");

        /**
         * @brief Builds an SPPF node from "filters", with optional "kernel", 5 when absent,
         *        "pools", 3 when absent, "shortcut", false when absent, "bn_epsilon" and
         *        "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape with C of at least two.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "kernel" is even.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_sppf_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
