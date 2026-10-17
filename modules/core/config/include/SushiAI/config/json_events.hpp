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
         * @param handler Receives the events; an event that returns false ends the read
         *                with success, and the handler keeps its own reason.
         * @return FileNotFound for a file that cannot be opened; InvalidFile, naming the
         *         path and the byte offset, for text that is no JSON document, and for a
         *         read that fails.
         */
        [[nodiscard]] Status read_json_events(const std::filesystem::path& path,
                                              JSONEvents& handler);

    } // namespace Config
} // namespace SushiAI
