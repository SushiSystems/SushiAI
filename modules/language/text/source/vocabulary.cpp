/**************************************************************************/
/* vocabulary.cpp                                                         */
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
 * @file vocabulary.cpp
 * @brief Implements the vocabulary of a byte-pair tokenizer.
 * @author Mustafa Garip
 */

#include <SushiAI/language/text/vocabulary.hpp>

#include <array>
#include <cstddef>
#include <utility>


#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>

#include "disk.hpp"
#include "pair_key.hpp"

namespace SushiAI::Language::Text
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_UNIT = "Vocabulary";

        /**
         * @brief Returns the names of every special token, end-of-text first.
         * @param extra The names after end-of-text, moved from.
         * @return The names; InvalidDescription for too many, or for one that is empty, too
         *         long or held twice.
         */
        [[nodiscard]] Result<std::vector<std::string>> checked_special_names(
            std::vector<std::string> extra)
        {
            SA_FAIL_IF(extra.size() + 1 > static_cast<std::size_t>(K_MAX_SPECIAL_TOKENS),
                       StatusCode::InvalidDescription,
                       "{}: {} special tokens with end-of-text; there may be at most {}", K_UNIT,
                       extra.size() + 1, K_MAX_SPECIAL_TOKENS);

            std::vector<std::string> names;
            names.reserve(extra.size() + 1);
            names.emplace_back(K_END_OF_TEXT_NAME);

            for (std::string& name : extra)
            {
                SA_FAIL_IF(name.empty() || name.size() > K_MAX_SPECIAL_NAME_BYTES,
                           StatusCode::InvalidDescription,
                           "{}: special token {} has a name of {} bytes; it has 1 to {}", K_UNIT,
                           names.size(), name.size(), K_MAX_SPECIAL_NAME_BYTES);

                for (const std::string& held : names)
                {
                    SA_FAIL_IF(held == name, StatusCode::InvalidDescription,
                               "{}: the special token '{}' is named twice", K_UNIT, name);
                }

                names.push_back(std::move(name));
            }

            return Result<std::vector<std::string>>(std::move(names));
        }
    } // namespace

    Result<Vocabulary> Vocabulary::make(std::vector<Merge> merges,
                                        std::vector<std::string> special_names) noexcept
    {
        return Detail::at_boundary(K_UNIT, [&]() -> Result<Vocabulary>
        {
            Vocabulary vocabulary;
            SA_TRY_ASSIGN(vocabulary.special_names_,
                          checked_special_names(std::move(special_names)));

            const TokenID first_special = K_BYTE_TOKEN_COUNT;
            const TokenID first_merged =
                first_special + static_cast<TokenID>(vocabulary.special_names_.size());
            const std::size_t room = static_cast<std::size_t>(K_MAX_VOCABULARY_SIZE - first_merged);

            SA_FAIL_IF(merges.size() > room, StatusCode::InvalidDescription,
                       "{}: {} merges after {} tokens; a vocabulary has at most {} tokens", K_UNIT,
                       merges.size(), first_merged, K_MAX_VOCABULARY_SIZE);

            std::string& bytes = vocabulary.bytes_;
            std::vector<std::uint32_t>& offsets = vocabulary.offsets_;
            offsets.reserve(static_cast<std::size_t>(first_merged) + merges.size() + 1);
            vocabulary.pairs_.reserve(merges.size());

            for (std::int32_t byte = 0; byte < K_BYTE_TOKEN_COUNT; ++byte)
            {
                offsets.push_back(static_cast<std::uint32_t>(bytes.size()));
                bytes.push_back(static_cast<char>(static_cast<unsigned char>(byte)));
            }

            // A special token stands for no byte, so its bytes begin where they end.
            offsets.insert(offsets.end(), vocabulary.special_names_.size() + 1,
                           static_cast<std::uint32_t>(bytes.size()));

            std::array<char, K_MAX_TOKEN_BYTES> joined;

            for (std::size_t i = 0; i < merges.size(); ++i)
            {
                const Merge merge = merges[i];
                const TokenID id = first_merged + static_cast<TokenID>(i);

                const auto is_part = [&](TokenID part) noexcept
                {
                    return part >= 0 && part < id &&
                           (part < first_special || part >= first_merged);
                };

                SA_FAIL_IF(!is_part(merge.left) || !is_part(merge.right),
                           StatusCode::InvalidDescription,
                           "{}: merge {} joins the tokens {} and {}; each must be a byte or a "
                           "merged token below {}",
                           K_UNIT, i, merge.left, merge.right, id);

                const std::string_view left = vocabulary.bytes_of(merge.left);
                const std::string_view right = vocabulary.bytes_of(merge.right);

                SA_FAIL_IF(left.size() + right.size() > K_MAX_TOKEN_BYTES,
                           StatusCode::InvalidDescription,
                           "{}: merge {} stands for {} bytes; a token has at most {}", K_UNIT, i,
                           left.size() + right.size(), K_MAX_TOKEN_BYTES);
                SA_FAIL_IF(!vocabulary.pairs_.emplace(Detail::pair_key(merge.left, merge.right), id)
                                .second,
                           StatusCode::InvalidDescription,
                           "{}: merge {} joins the tokens {} and {}, which an earlier merge joins",
                           K_UNIT, i, merge.left, merge.right);

                // The views point into bytes, which the append below may move.
                left.copy(joined.data(), left.size());
                right.copy(joined.data() + left.size(), right.size());
                bytes.append(joined.data(), left.size() + right.size());
                offsets.push_back(static_cast<std::uint32_t>(bytes.size()));
            }

            vocabulary.merges_ = std::move(merges);
            return Result<Vocabulary>(std::move(vocabulary));
        });
    }

    std::int32_t Vocabulary::size() const noexcept
    {
        return static_cast<std::int32_t>(offsets_.size()) - 1;
    }

    bool Vocabulary::contains(TokenID id) const noexcept
    {
        return id >= 0 && id < size();
    }

    TokenID Vocabulary::first_merged() const noexcept
    {
        return K_BYTE_TOKEN_COUNT + static_cast<TokenID>(special_names_.size());
    }

    TokenKind Vocabulary::kind_of(TokenID id) const noexcept
    {
        SA_FATAL_IF(!contains(id), "{}: the id {} is outside [0, {})", K_UNIT, id, size());

        if (id < K_BYTE_TOKEN_COUNT)
            return TokenKind::Byte;

        return id < first_merged() ? TokenKind::Special : TokenKind::Merged;
    }

    std::string_view Vocabulary::bytes_of(TokenID id) const noexcept
    {
        SA_FATAL_IF(!contains(id), "{}: the id {} is outside [0, {})", K_UNIT, id, size());

        const std::size_t begin = offsets_[static_cast<std::size_t>(id)];
        const std::size_t end = offsets_[static_cast<std::size_t>(id) + 1];
        return std::string_view(bytes_.data() + begin, end - begin);
    }

    std::optional<TokenID> Vocabulary::find_merge(TokenID left, TokenID right) const noexcept
    {
        if (left < 0 || right < 0)
            return std::nullopt;

        const auto found = pairs_.find(Detail::pair_key(left, right));
        if (found == pairs_.end())
            return std::nullopt;

        return found->second;
    }

    std::optional<TokenID> Vocabulary::find_special(std::string_view name) const noexcept
    {
        for (std::size_t i = 0; i < special_names_.size(); ++i)
        {
            if (special_names_[i] == name)
                return K_BYTE_TOKEN_COUNT + static_cast<TokenID>(i);
        }

        return std::nullopt;
    }

    const std::vector<Merge>& Vocabulary::merges() const noexcept
    {
        return merges_;
    }

    const std::vector<std::string>& Vocabulary::special_names() const noexcept
    {
        return special_names_;
    }
} // namespace SushiAI::Language::Text
