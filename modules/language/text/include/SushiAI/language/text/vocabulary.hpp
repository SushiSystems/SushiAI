/**************************************************************************/
/* vocabulary.hpp                                                         */
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
 * @file vocabulary.hpp
 * @brief Declares the vocabulary of a byte-pair tokenizer and the merge it is built from.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/text/token.hpp>

/** @brief Holds the tokens of a text, the vocabulary and the byte-pair tokenizer. */
namespace SushiAI::Language::Text
{
    /** @brief Names the two tokens one merged token stands for, in the order they are read. */
    struct Merge
    {
        TokenID left = 0;  /**< The token read first. */
        TokenID right = 0; /**< The token read after it. */
    };

    /** @brief Reports whether two merges name the same pair. */
    [[nodiscard]] constexpr bool operator==(const Merge& a, const Merge& b) noexcept
    {
        return a.left == b.left && a.right == b.right;
    }

    /** @brief Reports whether two merges name different pairs. */
    [[nodiscard]] constexpr bool operator!=(const Merge& a, const Merge& b) noexcept
    {
        return !(a == b);
    }

    /** @brief Names what a token stands for. */
    enum class TokenKind : std::uint8_t
    {
        Byte,    /**< One byte; ids [0, 256). */
        Special, /**< No byte of a text; ids [256, 256 + special count). */
        Merged,  /**< The bytes of two earlier tokens; the ids after the special tokens. */
    };

    /**
     * @brief Holds 256 byte tokens, the special tokens and the merged tokens, in id order.
     *
     * Merge i has the id first_merged() + i, so a smaller id was learned earlier. No method
     * changes a vocabulary.
     * @see modules/language/text/README.md, section "Token ids".
     */
    class Vocabulary final
    {
        public:
            /**
             * @brief Builds a vocabulary from its merges and its special tokens.
             * @param merges Each names two byte or earlier merged tokens, and no pair twice.
             * @param special_names The special tokens after end-of-text; unique, not empty.
             * @return The vocabulary; InvalidDescription for a merge or a name that breaks a
             *         rule above or a limit of token.hpp, naming it.
             */
            [[nodiscard]] static Result<Vocabulary> make(
                std::vector<Merge> merges, std::vector<std::string> special_names = {});

            /** @brief Returns the token count; every id is in [0, size()). */
            [[nodiscard]] std::int32_t size() const noexcept;

            /** @brief Reports whether an id names a token of this vocabulary. */
            [[nodiscard]] bool contains(TokenID id) const noexcept;

            /** @brief Returns the id of the first merged token, or size() when there is none. */
            [[nodiscard]] TokenID first_merged() const noexcept;

            /**
             * @brief Returns what a token stands for.
             * @pre contains(id); an id that is not stops the process.
             */
            [[nodiscard]] TokenKind kind_of(TokenID id) const noexcept;

            /**
             * @brief Returns the bytes a token stands for; none for a special token.
             * @return A view that lives as long as this vocabulary.
             * @pre contains(id); an id that is not stops the process.
             */
            [[nodiscard]] std::string_view bytes_of(TokenID id) const noexcept;

            /** @brief Returns the merged token of a pair, or nothing when no merge names it. */
            [[nodiscard]] std::optional<TokenID> find_merge(TokenID left,
                                                            TokenID right) const noexcept;

            /** @brief Returns the special token of a name, or nothing when none has it. */
            [[nodiscard]] std::optional<TokenID> find_special(
                std::string_view name) const noexcept;

            /** @brief Returns the merges in the order they were learned. */
            [[nodiscard]] const std::vector<Merge>& merges() const noexcept;

            /** @brief Returns the names of the special tokens in id order, end-of-text first. */
            [[nodiscard]] const std::vector<std::string>& special_names() const noexcept;

        private:
            /** @brief Builds nothing; make() fills the members. */
            Vocabulary() = default;

            /// @brief Holds the name of each special token, in id order.
            std::vector<std::string> special_names_;

            /// @brief Holds the pair of each merged token, in id order.
            std::vector<Merge> merges_;

            /// @brief Holds the bytes of every token one after another, in id order.
            std::string bytes_;

            /// @brief Holds where each token's bytes begin in bytes_, and their end last.
            std::vector<std::uint32_t> offsets_;

            /// @brief Maps a pair, left in the high word, to its merged token.
            std::unordered_map<std::uint64_t, TokenID> pairs_;
    };
} // namespace SushiAI::Language::Text
