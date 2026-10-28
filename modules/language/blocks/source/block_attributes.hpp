/**************************************************************************/
/* block_attributes.hpp                                                   */
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
 * @file block_attributes.hpp
 * @brief Defines the keys and the defaults the block rows share and names the readers they use.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_attributes.hpp>

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Names the key that holds a row's head count. */
            inline constexpr std::string_view K_HEADS_KEY = "heads";

            /** @brief Names the key that holds the width of a row's feed-forward layer. */
            inline constexpr std::string_view K_HIDDEN_KEY = "hidden";

            /** @brief Names the key that says whether a row's attention is masked. */
            inline constexpr std::string_view K_CAUSAL_KEY = "causal";

            /** @brief Names the key that says where a row's attention takes positions from. */
            inline constexpr std::string_view K_POSITIONS_KEY = "positions";

            /** @brief Names the key that holds a row's vocabulary size. */
            inline constexpr std::string_view K_VOCABULARY_KEY = "vocabulary";

            /** @brief Names the key that holds the row count of a position table. */
            inline constexpr std::string_view K_LENGTH_KEY = "length";

            /** @brief Names the key that holds the embedding row a head scores with. */
            inline constexpr std::string_view K_TIE_KEY = "tie";

            /** @brief Names the key that holds the expert count of a row. */
            inline constexpr std::string_view K_EXPERTS_KEY = "experts";

            /** @brief Names the key that holds how many experts a position keeps. */
            inline constexpr std::string_view K_TOP_K_KEY = "top_k";

            /** @brief Holds the top_k of a row that does not say: every expert is kept. */
            inline constexpr int64_t K_DEFAULT_TOP_K = 0;

            /** @brief Holds the expert count of a transformer row that states none. */
            inline constexpr int64_t K_NO_EXPERTS = 0;

            /** @brief Holds whether attention is masked in a row that does not say. */
            inline constexpr bool K_DEFAULT_CAUSAL = true;

            /** @brief Holds hidden / dim of a transformer row that states no hidden width. */
            inline constexpr int64_t K_DEFAULT_HIDDEN_RATIO = 4;

            /** @brief Names the rank of a per-sample [T, C] sequence. */
            inline constexpr int32_t K_SAMPLE_RANK = 2;

            using NN::Detail::K_LAYER_NORM_EPSILON_KEY;
            using NN::Detail::KeySpan;
            using NN::Detail::read_bool_or;
            using NN::Detail::read_layer_norm_epsilon;
            using NN::Detail::read_positive_int;
            using NN::Detail::read_positive_int_or;
            using NN::Detail::read_row_above;
            using NN::Detail::reject_unknown_keys;
            using NN::Detail::wrap_block;

            /**
             * @brief Returns the one [T, C] shape a block row reads.
             * @return InvalidDescription unless @p inputs holds exactly one rank-2 shape.
             */
            [[nodiscard]] inline Result<Shape> read_sequence(NN::ShapeSpan inputs,
                                                             std::string_view context)
            {
                SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads one input, but it was given {}", context, inputs.size());
                SA_FAIL_IF(inputs[0].rank() != K_SAMPLE_RANK, StatusCode::InvalidDescription,
                           "{} was given a per-sample input of {}, but it reads a rank-2 "
                           "[T, C] input",
                           context, inputs[0].to_string());

                return inputs[0];
            }

            /**
             * @brief Reads the whole number under @p key, or @p fallback when the key is absent.
             * @return InvalidDescription when the key holds anything but a whole number of
             *         zero or more.
             */
            [[nodiscard]] inline Result<int64_t> read_count_or(const Config::Value& attributes,
                                                               std::string_view key,
                                                               std::string_view context,
                                                               int64_t fallback)
            {
                if (!attributes.has(key))
                    return fallback;

                SA_TRY_ASSIGN(const Config::Value& value, attributes.at(key, context));
                SA_FAIL_IF(value.kind() != Config::Kind::Int, StatusCode::InvalidDescription,
                           "{} '{}' must be a whole number, but it is {}", context, key,
                           Config::kind_name(value.kind()));

                SA_TRY_ASSIGN(const int64_t number, value.as_int(key));
                SA_FAIL_IF(number < 0, StatusCode::InvalidDescription,
                           "{} '{}' must be zero or more, but it is {}", context, key, number);

                return number;
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
