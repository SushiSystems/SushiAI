/**************************************************************************/
/* codec.cpp                                                              */
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
 * @file codec.cpp
 * @brief Implements the encoder from text bytes to token ids and the decoder back.
 * @author Mustafa Garip
 */

#include <SushiAI/language/text/codec.hpp>

#include <array>
#include <cstddef>
#include <optional>


#include "chunks.hpp"
#include "disk.hpp"

namespace SushiAI::Language::Text
{
    namespace
    {
        /** @brief Names the encoder in a message. */
        constexpr std::string_view K_ENCODER = "encode_text";

        /** @brief Names the decoder in a message. */
        constexpr std::string_view K_DECODER = "decode_tokens";

        /** @brief Holds the tokens of one chunk while its merges are applied. */
        using ChunkTokens = std::array<TokenID, Detail::K_MAX_CHUNK_BYTES>;

        /**
         * @brief Returns the merged token learned first among the adjacent pairs of a chunk.
         * @return The token, or nothing when no adjacent pair has a merge.
         */
        [[nodiscard]] std::optional<TokenID> earliest_merge(const Vocabulary& vocabulary,
                                                            const ChunkTokens& tokens,
                                                            std::size_t count) noexcept
        {
            std::optional<TokenID> earliest;

            for (std::size_t i = 0; i + 1 < count; ++i)
            {
                const std::optional<TokenID> merged =
                    vocabulary.find_merge(tokens[i], tokens[i + 1]);

                if (merged.has_value() && (!earliest.has_value() || *merged < *earliest))
                    earliest = merged;
            }

            return earliest;
        }

        /**
         * @brief Replaces each occurrence of a pair with its merged token, from the left.
         * @return The token count after the replacement.
         */
        [[nodiscard]] std::size_t apply_merge(ChunkTokens& tokens, std::size_t count, Merge merge,
                                              TokenID merged) noexcept
        {
            std::size_t out = 0;

            for (std::size_t i = 0; i < count;)
            {
                const bool is_pair =
                    i + 1 < count && tokens[i] == merge.left && tokens[i + 1] == merge.right;

                tokens[out++] = is_pair ? merged : tokens[i];
                i += is_pair ? 2 : 1;
            }

            return out;
        }

        /** @brief Appends the tokens of one chunk of at most K_MAX_CHUNK_BYTES bytes. */
        void encode_chunk(const Vocabulary& vocabulary, std::string_view chunk,
                          std::vector<TokenID>& ids)
        {
            ChunkTokens tokens;
            std::size_t count = chunk.size();

            for (std::size_t i = 0; i < count; ++i)
                tokens[i] = static_cast<TokenID>(static_cast<unsigned char>(chunk[i]));

            // A merge makes pairs of its own token alone, and those were learned after it.
            for (std::optional<TokenID> merged = earliest_merge(vocabulary, tokens, count);
                 merged.has_value(); merged = earliest_merge(vocabulary, tokens, count))
            {
                const TokenID rank = *merged - vocabulary.first_merged();
                count = apply_merge(tokens, count,
                                    vocabulary.merges()[static_cast<std::size_t>(rank)], *merged);
            }

            ids.insert(ids.end(), tokens.begin(), tokens.begin() + count);
        }
    } // namespace

    Status encode_text(const Vocabulary& vocabulary, std::string_view text,
                       std::vector<TokenID>& ids)
    {
        return Detail::at_boundary(K_ENCODER, [&]() -> Status
        {
            for (std::size_t begin = 0; begin < text.size();)
            {
                const std::size_t end = Detail::chunk_end(text, begin);
                encode_chunk(vocabulary, text.substr(begin, end - begin), ids);
                begin = end;
            }

            return Status();
        });
    }

    Status decode_tokens(const Vocabulary& vocabulary, SushiRuntime::span<const TokenID> ids,
                         std::string& text)
    {
        return Detail::at_boundary(K_DECODER, [&]() -> Status
        {
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                SA_FAIL_IF(!vocabulary.contains(ids[i]), StatusCode::InvalidDescription,
                           "{}: the id {} at index {} is outside [0, {})", K_DECODER, ids[i], i,
                           vocabulary.size());

                text.append(vocabulary.bytes_of(ids[i]));
            }

            return Status();
        });
    }
} // namespace SushiAI::Language::Text
