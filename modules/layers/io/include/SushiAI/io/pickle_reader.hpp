/**************************************************************************/
/* pickle_reader.hpp                                                      */
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
 * @file pickle_reader.hpp
 * @brief Declares the value tree of a pickle stream and the reader that builds it, running no code.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Names what one node of a pickle value tree holds. */
        enum class PickleKind : std::uint8_t
        {
            None,         /**< Python's None. */
            Bool,         /**< True or False, in boolean. */
            Int,          /**< A whole number of 64 bits, in integer. */
            Float,        /**< A double, in real. */
            String,       /**< A text, in text. */
            Bytes,        /**< A byte string, in text. */
            List,         /**< A list, its elements in items. */
            Tuple,        /**< A tuple, its elements in items. */
            Set,          /**< A set or a frozen set, its elements in items. */
            Dict,         /**< A dictionary, its pairs in entries. */
            Global,       /**< A name the stream imports, as "module name", in text. */
            Call,         /**< A call the stream asks for and the reader did not make. */
            PersistentID, /**< A reference to data outside the stream, its id in items[0]. */
        };

        struct PickleValue;

        /** @brief Holds one shared node of a pickle tree; the memo can name one node twice. */
        using PickleRef = std::shared_ptr<PickleValue>;

        /**
         * @struct PickleValue
         * @brief Holds one node of the tree a pickle stream describes.
         *
         * A Call holds its callable, a Global, and its arguments in items. What the stream then
         * builds onto the result lands beside them: BUILD in state, SETITEMS in entries and
         * APPENDS in appended. Nothing is ever called.
         */
        struct PickleValue
        {
            /** @brief Holds what this node is. */
            PickleKind kind = PickleKind::None;

            /** @brief Holds the value of a Bool. */
            bool boolean = false;

            /** @brief Holds the value of an Int. */
            std::int64_t integer = 0;

            /** @brief Holds the value of a Float. */
            double real = 0.0;

            /** @brief Holds a String, the bytes of a Bytes, or the "module name" of a Global. */
            std::string text;

            /** @brief Holds a sequence's elements, a Call's arguments or a persistent id. */
            std::vector<PickleRef> items;

            /** @brief Holds a Dict's pairs, or the pairs set onto a Call, in stream order. */
            std::vector<std::pair<PickleRef, PickleRef>> entries;

            /** @brief Holds the elements appended onto a Call. */
            std::vector<PickleRef> appended;

            /** @brief Holds the Global a Call names. */
            PickleRef callable;

            /** @brief Holds what BUILD gave a Call, or null when it was given nothing. */
            PickleRef state;
        };

        /**
         * @brief Returns whether @p value is a Call of the global "module name" @p name.
         * @param value May be null.
         */
        [[nodiscard]] bool is_call_of(const PickleValue* value, std::string_view name) noexcept;

        /**
         * @brief Returns the value paired with the String key @p key in a Dict or a Call's
         *        entries, or null when no pair holds it.
         */
        [[nodiscard]] const PickleValue* find_entry(const PickleValue& value,
                                                    std::string_view key) noexcept;

        /**
         * @brief Reads a pickle stream of protocol 2 to 5 into its value tree.
         * @param accepted The globals the stream may name: a whole "module name", or a module
         *                 prefix ending in '.' that accepts every name of such a module.
         * @return InvalidFile for a stream that is truncated or malformed, naming the byte;
         *         Unsupported for an opcode the reader lacks or a global it does not accept.
         */
        [[nodiscard]] Result<PickleRef> read_pickle(
            SushiRuntime::span<const std::byte> bytes,
            SushiRuntime::span<const std::string_view> accepted, std::string_view origin) noexcept;
    } // namespace IO
} // namespace SushiAI
