/**************************************************************************/
/* c3k2.hpp                                                               */
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
 * @file c3k2.hpp
 * @brief Declares C3k2, a split stem with a chain of inner blocks, and the table of inner kinds.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/psa_block.hpp>

namespace SushiAI
{
    namespace Vision::Blocks
    {
        /**
         * @brief Splits a stem in two, chains inner blocks off the second half, joins all parts.
         * @tparam Inner A Traceable block that maps the hidden width to the hidden width.
         */
        template<typename Inner>
        struct C3k2
        {
            /** @brief Holds the width of each half of the stem and of every inner block. */
            int64_t hidden = 0;

            /** @brief Holds the 1x1 convolution that makes the stem, 2 * hidden wide. */
            Conv cv1;

            /** @brief Holds the 1x1 convolution over the two halves and every inner result. */
            Conv cv2;

            /** @brief Holds the inner blocks, in the order they are chained. */
            NN::ModuleList<Inner> m;

            /**
             * @brief Builds the two convolutions around the inner blocks given.
             * @param inner The inner blocks, each mapping hidden channels to hidden channels.
             */
            C3k2(int64_t in_channels, int64_t out_channels, int64_t hidden_channels,
                 NN::ModuleList<Inner> inner, NN::NormOptions norm)
                : hidden(hidden_channels),
                  cv1(in_channels, 2 * hidden_channels, 1, 1, norm),
                  cv2((2 + static_cast<int64_t>(inner.size())) * hidden_channels, out_channels,
                      1, 1, norm),
                  m(std::move(inner))
            {

            }

            /** @brief Lists the three children, under "cv1", "cv2" and "m". */
            static constexpr auto fields()
            {
                return std::tuple{
                    NN::child("cv1", &C3k2::cv1),
                    NN::child("cv2", &C3k2::cv2),
                    NN::child("m", &C3k2::m),
                };
            }

            /**
             * @brief Traces cv1, the two slices of its result, the chain, then cv2 over all parts.
             * @param x An NHWC [N, H, W, in_channels] value.
             * @return The [N, H, W, out_channels] result's id.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                const Graph::ValueID stem = NN::trace_layer(cv1, trace, x);

                std::vector<Graph::ValueID> parts;
                parts.reserve(2 + m.size());
                parts.push_back(builder.slice_channels(stem, 0, hidden));
                parts.push_back(builder.slice_channels(stem, hidden, hidden));
                for (Inner& block : m)
                    parts.push_back(NN::trace_layer(block, trace, parts.back()));

                return NN::trace_layer(cv2, trace,
                                       builder.concat_channels(
                                           SushiRuntime::span<const Graph::ValueID>(
                                               parts.data(), parts.size())));
            }

            /**
             * @brief Computes the per-sample shape forward() produces.
             * @param input A per-sample [H, W, in_channels] shape.
             */
            Shape output_shape(const Shape& input) const
            {
                // cv1 refuses an input that is not [H, W, in_channels] and keeps H and W.
                const Shape stem = cv1.output_shape(input);

                return cv2.output_shape(Shape{stem[0], stem[1], cv2.conv.in_channels});
            }
        };

        static_assert(NN::TraceModule<C3k2<Bottleneck>>,
                      "C3k2 over Bottleneck must satisfy the TraceModule contract");
        static_assert(NN::TraceModule<C3k2<C3k>>,
                      "C3k2 over C3k must satisfy the TraceModule contract");
        static_assert(NN::TraceModule<C3k2<NN::Sequential<Bottleneck, PSABlock>>>,
                      "C3k2 over Bottleneck and PSABlock must satisfy the TraceModule contract");

        /** @brief Holds one row of K_C3K2_INNER. */
        struct C3k2InnerEntry
        {
            /** @brief Holds the "inner" a c3k2 row spells. */
            std::string_view name;

            /** @brief Points at the factory that builds a C3k2 over that inner block. */
            NN::NodeFactory make;
        };

        /**
         * @brief Builds a C3k2 node whose inner blocks are Bottleneck(hidden, hidden, hidden / 2).
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @param attributes The keys make_c3k2_node() documents.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "inner" is stated and is not "bottleneck".
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k2_of_bottleneck(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

        /**
         * @brief Builds a C3k2 node whose inner blocks are C3k(hidden, hidden) of two bottlenecks.
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @param attributes The keys make_c3k2_node() documents.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "inner" is stated and is not "c3k".
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k2_of_c3k(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

        /**
         * @brief Builds a C3k2 node whose inner blocks are Bottleneck(hidden, hidden, hidden / 2)
         *        and PSABlock(hidden, max(hidden / 64, 1)).
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @param attributes The keys make_c3k2_node() documents.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, when the
         *         head count does not split "hidden", or when "inner" is not "bottleneck_psa".
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k2_of_bottleneck_psa(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

        /** @brief Lists every inner block a c3k2 row may name. */
        inline constexpr std::array<C3k2InnerEntry, 3> K_C3K2_INNER = {
            C3k2InnerEntry{"bottleneck", &make_c3k2_of_bottleneck},
            C3k2InnerEntry{"c3k", &make_c3k2_of_c3k},
            C3k2InnerEntry{"bottleneck_psa", &make_c3k2_of_bottleneck_psa},
        };

        /**
         * @brief Builds a C3k2 node from "filters" and "hidden", with optional "inner",
         *        "bottleneck" when absent, "repeats", 1 when absent, "shortcut", true when
         *        absent, "bn_epsilon" and "bn_momentum".
         * @param inputs One per-sample [H, W, C] shape; C becomes the input width.
         * @return The node, or InvalidDescription when @p inputs or a key does not fit, or
         *         "inner" names no row of K_C3K2_INNER.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_c3k2_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace Vision::Blocks
} // namespace SushiAI
