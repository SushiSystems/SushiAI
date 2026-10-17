/**************************************************************************/
/* positions_attribute.hpp                                                */
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
 * @file positions_attribute.hpp
 * @brief Defines the reader of the "positions" key of a row that holds an attention.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Pairs what a file writes under "positions" with what it selects. */
            struct PositionsName
            {
                std::string_view name;        /**< @brief The word of the file. */
                AttentionPositions positions; /**< @brief What the word selects. */
            };

            /** @brief Lists every word "positions" accepts; the first is the default. */
            inline constexpr PositionsName K_POSITIONS_NAMES[] = {
                {"learned", AttentionPositions::Learned},
                {"rotary", AttentionPositions::Rotary},
            };

            /** @brief Returns every word of K_POSITIONS_NAMES, quoted, for a diagnostic. */
            [[nodiscard]] inline std::string list_positions()
            {
                std::string listed;
                for (const PositionsName& row : K_POSITIONS_NAMES)
                    listed.append(listed.empty() ? "'" : ", '").append(row.name).append("'");

                return listed;
            }

            /**
             * @brief Reads "positions", or returns the first row of K_POSITIONS_NAMES when
             *        the key is absent.
             * @return InvalidDescription for a value that is no string or no listed word.
             */
            [[nodiscard]] inline Result<AttentionPositions> read_positions(
                const Config::Value& attributes, std::string_view context)
            {
                if (!attributes.has(K_POSITIONS_KEY))
                    return K_POSITIONS_NAMES[0].positions;

                const Config::Value& value = attributes.at(K_POSITIONS_KEY, context);
                SA_FAIL_IF(value.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                           "{} '{}' must be a string, but it is {}", context, K_POSITIONS_KEY,
                           Config::kind_name(value.kind()));

                const std::string& word = value.as_string(K_POSITIONS_KEY);
                const PositionsName* found = nullptr;
                for (const PositionsName& row : K_POSITIONS_NAMES)
                {
                    if (row.name == word)
                        found = &row;
                }

                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "{} '{}' is '{}'; it is one of {}", context, K_POSITIONS_KEY, word,
                           list_positions());

                return found->positions;
            }

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
