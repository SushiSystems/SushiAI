/**************************************************************************/
/* json_events.hpp                                                        */
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
 * @file json_events.hpp
 * @brief Declares the events of a JSON document and the reader that streams them from a file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Holds the bytes of a file read_json_events() keeps in memory at once. */
        inline constexpr std::size_t K_JSON_EVENT_BUFFER_BYTES = 65536;

        /**
         * @class JSONEvents
         * @brief Receives a JSON document as events, in the order the text spells them.
         *
         * Every event returns whether the read goes on; false ends it at that event.
         */
        class JSONEvents
        {
            public:
                virtual ~JSONEvents() = default;

                /** @brief Receives the opening of an object. */
                [[nodiscard]] virtual bool begin_object() = 0;

                /** @brief Receives the closing of the object opened last. */
                [[nodiscard]] virtual bool end_object() = 0;

                /** @brief Receives the opening of an array. */
                [[nodiscard]] virtual bool begin_array() = 0;

                /** @brief Receives the closing of the array opened last. */
                [[nodiscard]] virtual bool end_array() = 0;

                /**
                 * @brief Receives a key of the object opened last; its value is the next event.
                 * @param text Valid during the call only.
                 */
                [[nodiscard]] virtual bool key(std::string_view text) = 0;

                /** @brief Receives a null. */
                [[nodiscard]] virtual bool null() = 0;

                /** @brief Receives true or false. */
                [[nodiscard]] virtual bool boolean(bool value) = 0;

                /** @brief Receives a number spelled without fraction or exponent. */
                [[nodiscard]] virtual bool whole(std::int64_t value) = 0;

                /**
                 * @brief Receives a number with a fraction or an exponent, or a whole number
                 *        above the largest std::int64_t.
                 */
                [[nodiscard]] virtual bool real(double value) = 0;

                /**
                 * @brief Receives a string.
                 * @param text Valid during the call only.
                 */
                [[nodiscard]] virtual bool string(std::string_view text) = 0;
        };

        /**
         * @brief Streams the JSON document at @p path to @p handler, one event at a time.
         * @param handler Receives the events; false from an event ends the read with success.
         * @param buffer_bytes Bytes of the file held in memory at once; must be positive.
         * @return FileNotFound for a file that cannot be opened; InvalidFile, naming the
         *         path and the byte offset, for text that is no JSON, and for a failed read.
         */
        [[nodiscard]] Status read_json_events(
            const std::filesystem::path& path, JSONEvents& handler,
            std::size_t buffer_bytes = K_JSON_EVENT_BUFFER_BYTES) noexcept;

        /**
         * @brief Hands the JSON document in @p text to @p handler, one event at a time.
         * @param text The whole document, held by the caller for the call.
         * @param name Names the document in a failure, as a path would.
         * @param handler Receives the events, as read_json_events() hands them.
         * @return InvalidFile, naming @p name and the byte offset, for text that is no JSON
         *         document.
         */
        [[nodiscard]] Status read_json_text_events(std::string_view text, std::string_view name,
                                                   JSONEvents& handler) noexcept;

    } // namespace Config
} // namespace SushiAI
