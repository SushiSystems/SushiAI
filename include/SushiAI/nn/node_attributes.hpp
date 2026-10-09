/**************************************************************************/
/* node_attributes.hpp                                                    */
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
 * @file node_attributes.hpp
 * @brief Declares the status-returning readers a node factory reads a model-file row with.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace Detail
        {
            /** @brief Names the key that holds a row's batch-normalisation epsilon. */
            inline constexpr std::string_view K_NORM_EPSILON_KEY = "bn_epsilon";

            /** @brief Names the key that holds a row's batch-normalisation momentum. */
            inline constexpr std::string_view K_NORM_MOMENTUM_KEY = "bn_momentum";

            /** @brief Views the keys one node row accepts. */
            using KeySpan = SushiRuntime::span<const std::string_view>;

            /** @brief Joins @p names into "a, b, c" for a diagnostic. */
            [[nodiscard]] std::string list_names(KeySpan names);

            /**
             * @brief Returns the one [H, W, C] shape a node row reads.
             * @return InvalidDescription unless @p inputs holds exactly one rank-3 shape.
             */
            [[nodiscard]] Result<Shape> read_feature_map(NN::ShapeSpan inputs,
                                                         std::string_view context);

            /** @brief Returns InvalidDescription when @p attributes has a key not in @p allowed. */
            [[nodiscard]] Status reject_unknown_keys(const Config::Value& attributes,
                                                     KeySpan allowed, std::string_view context);

            /**
             * @brief Reads the required whole number under @p key.
             * @return InvalidDescription when the key is absent, not a whole number or below one.
             */
            [[nodiscard]] Result<int64_t> read_positive_int(const Config::Value& attributes,
                                                            std::string_view key,
                                                            std::string_view context);

            /**
             * @brief Reads the whole number under @p key, or @p fallback when the key is absent.
             * @return InvalidDescription when the key holds anything but a whole number above zero.
             */
            [[nodiscard]] Result<int64_t> read_positive_int_or(const Config::Value& attributes,
                                                               std::string_view key,
                                                               std::string_view context,
                                                               int64_t fallback);

            /**
             * @brief Reads the boolean under @p key, or @p fallback when the key is absent.
             * @return InvalidDescription when the key holds anything but a boolean.
             */
            [[nodiscard]] Result<bool> read_bool_or(const Config::Value& attributes,
                                                    std::string_view key,
                                                    std::string_view context, bool fallback);

            /**
             * @brief Reads K_NORM_EPSILON_KEY and K_NORM_MOMENTUM_KEY over NormOptions' defaults.
             * @return InvalidDescription unless epsilon is positive and momentum is in [0, 1].
             */
            [[nodiscard]] Result<NN::NormOptions> read_norm_options(
                const Config::Value& attributes, std::string_view context);

            /**
             * @brief Wraps @p block as the one-port node of a model graph.
             * @param input The per-sample shape the node reads.
             * @return The node, or the failure of the block's output_shape() over @p input.
             */
            template<typename Block>
            [[nodiscard]] Result<std::unique_ptr<NN::Node>> wrap_block(Block block,
                                                                       const Shape& input)
            {
                SA_TRY_ASSIGN(std::unique_ptr<NN::ModuleNode<Block>> node,
                              NN::ModuleNode<Block>::create(std::move(block), input));

                return std::unique_ptr<NN::Node>(std::move(node));
            }

        } // namespace Detail
    } // namespace NN
} // namespace SushiAI
