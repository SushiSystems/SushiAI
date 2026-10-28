/**************************************************************************/
/* language_head.hpp                                                      */
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
 * @file language_head.hpp
 * @brief Declares LanguageHead, the final normalisation and the linear layer to the vocabulary.
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
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Normalises a [N, T, dim] sequence and maps each position to vocabulary scores. */
        struct LanguageHead
        {
            /** @brief Holds the width of the input. */
            int64_t dim = 0;

            /** @brief Holds the number of tokens, which is the last extent of the result. */
            int64_t vocabulary = 0;

            /** @brief Holds the normalisation the linear layer reads. */
            NN::LayerNorm norm{};

            /** @brief Holds the linear layer from dim to vocabulary, which has no bias. */
            NN::Linear linear{};

            /** @brief Points at the table tie_to() was given; null for linear's own weight. */
            NN::Parameter* tied_weight = nullptr;

            /** @brief Builds a head of no width, which no input fits. */
            LanguageHead() = default;

            /** @brief Builds the two children; forward() checks the widths. */
            LanguageHead(int64_t dim, int64_t vocabulary, double epsilon) noexcept;

            /**
             * @brief Scores with @p table, a [vocabulary, dim] parameter another layer owns.
             * @param table An embedding's weight, traced before this head; it must outlive
             *        the head and stay at its address, so a copied or moved model ties anew.
             */
            void tie_to(NN::Parameter& table) noexcept;

            /** @brief Lists the two children, under "norm" and "linear". */
            static constexpr auto fields() noexcept
            {
                return std::tuple{
                    NN::child("norm", &LanguageHead::norm),
                    NN::child("linear", &LanguageHead::linear),
                };
            }

            /**
             * @brief Traces norm, then linear or the tied table over the [N * T, dim] rows.
             * @param x A [N, T, dim] value of a real floating type.
             * @return The [N, T, vocabulary] logits' id, or @p x after trace.fail() when @p x,
             *         this head's description or the tied table does not fit.
             */
            Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x) noexcept;

            /**
             * @brief Traces forward() over the newest slice, as a position reads no other.
             * @param sample Not read.
             */
            Graph::ValueID forward_incremental(const NN::Trace& trace, Graph::ValueID x,
                                               const Shape& sample) noexcept;

            /**
             * @brief Returns [T, vocabulary] for a per-sample [T, dim] shape.
             * @pre @p input has rank 2; another rank is fatal.
             */
            Result<Shape> output_shape(const Shape& input) const noexcept;
        };

        static_assert(NN::TraceModule<LanguageHead>,
                      "LanguageHead must satisfy the TraceModule contract");
        static_assert(NN::IncrementalModule<LanguageHead>,
                      "LanguageHead must have an incremental form");

        /**
         * @brief Builds a LanguageHead node from "vocabulary", with optional "ln_epsilon" and
         *        "tie", the "embedding" row above whose table the head scores with.
         * @param inputs One per-sample [T, C] shape; C becomes dim.
         * @param above The rows "tie" may name; a tied head leaves "vocabulary" optional.
         * @return The node, or InvalidDescription when @p inputs, a key or the tied row does
         *         not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<NN::Node>> make_language_head_node(
            NN::ShapeSpan inputs, const Config::Value& attributes, std::string_view context,
            const NN::BuiltRows& above = {}) noexcept;

    } // namespace Language::Blocks
} // namespace SushiAI
