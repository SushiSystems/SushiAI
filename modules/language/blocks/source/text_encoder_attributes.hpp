/**************************************************************************/
/* text_encoder_attributes.hpp                                            */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file text_encoder_attributes.hpp
 * @brief Declares the keys a text encoder reads and the reader that builds it from them.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/blocks/text_encoder.hpp>
#include "block_attributes.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace Detail
        {
            /** @brief Lists every key a "text_encoder" row accepts. */
            inline constexpr std::array<std::string_view, 7> K_ENCODER_KEYS = {
                K_VOCABULARY_KEY, K_DIM_KEY,    K_HEADS_KEY,
                K_BLOCKS_KEY,     K_HIDDEN_KEY, K_LENGTH_KEY,
                K_LAYER_NORM_EPSILON_KEY};

            /** @brief Views the keys of @p keys; @p keys must outlive the view. */
            template<std::size_t Count>
            [[nodiscard]] constexpr KeySpan keys_of(const std::array<std::string_view, Count>& keys)
                noexcept
            {
                return KeySpan(keys.data(), keys.size());
            }

            /**
             * @brief Returns @p keys followed by @p extra, for a row that reads one key more
             *        than the encoder it holds.
             */
            template<std::size_t Count>
            [[nodiscard]] constexpr std::array<std::string_view, Count + 1> keys_with(
                const std::array<std::string_view, Count>& keys, std::string_view extra) noexcept
            {
                std::array<std::string_view, Count + 1> joined{};
                for (std::size_t i = 0; i < Count; ++i)
                    joined[i] = keys[i];

                joined[Count] = extra;
                return joined;
            }

            /**
             * @brief Builds the encoder the keys of K_ENCODER_KEYS describe.
             * @param positions The tokens of the longest text the row reads; "length" defaults
             *        to it and may not fall below it.
             * @return The encoder, or InvalidDescription when a key is absent or misfits.
             */
            [[nodiscard]] Result<TextEncoder> read_text_encoder(const Config::Value& attributes,
                                                                int64_t positions,
                                                                std::string_view context) noexcept;

        } // namespace Detail
    } // namespace Language::Blocks
} // namespace SushiAI
