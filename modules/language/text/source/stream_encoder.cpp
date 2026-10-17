/**************************************************************************/
/* stream_encoder.cpp                                                     */
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
 * @file stream_encoder.cpp
 * @brief Implements encode_stream over pieces that end where a chunk ends.
 * @author Mustafa Garip
 */

#include <SushiAI/language/text/stream_encoder.hpp>

#include <string>
#include <string_view>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/language/text/codec.hpp>
#include "chunks.hpp"
#include "disk.hpp"

namespace SushiAI::Language::Text
{
    namespace
    {
        /** @brief Names the encoder in a message. */
        constexpr std::string_view K_STREAM_ENCODER = "encode_stream";

        /**
         * @brief Returns the last offset of @p text at which a blank byte follows another
         *        byte, which is where a chunk ends whatever came before it.
         * @return The offset of the blank byte, or 0 when @p text has no such place.
         */
        [[nodiscard]] std::size_t last_chunk_edge(std::string_view text) noexcept
        {
            for (std::size_t at = text.size(); at > 1; --at)
            {
                if (Detail::is_blank(text[at - 1]) && !Detail::is_blank(text[at - 2]))
                    return at - 1;
            }

            return 0;
        }
    } // namespace

    Status encode_stream(const Vocabulary& vocabulary, std::istream& in, std::size_t piece_bytes,
                         std::vector<TokenID>& ids)
    {
        SA_FATAL_IF(piece_bytes == 0, "{} was asked to read 0 bytes at a time", K_STREAM_ENCODER);

        return Detail::at_boundary(K_STREAM_ENCODER, [&]() -> Status
        {
            std::string held;
            std::string piece(piece_bytes, '\0');
            while (in.read(piece.data(), static_cast<std::streamsize>(piece.size())) ||
                   in.gcount() > 0)
            {
                held.append(piece.data(), static_cast<std::size_t>(in.gcount()));

                // Before the edge every chunk is whole; the rest waits for the next piece.
                const std::size_t edge = last_chunk_edge(held);
                SA_TRY(encode_text(vocabulary, std::string_view(held).substr(0, edge), ids));
                held.erase(0, edge);
            }

            SA_FAIL_IF(in.bad(), StatusCode::InvalidFile,
                       "{}: the stream failed after {} tokens were read", K_STREAM_ENCODER,
                       ids.size());

            return encode_text(vocabulary, held, ids);
        });
    }

} // namespace SushiAI::Language::Text
