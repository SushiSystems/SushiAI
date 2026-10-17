/**************************************************************************/
/* section_reader.hpp                                                     */
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
 * @file section_reader.hpp
 * @brief Declares the readers of one object of a configuration document that return a status.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief Views the keys a section accepts, or any other list of names. */
        using KeySpan = SushiRuntime::span<const std::string_view>;

        /** @brief Returns the text that names the member @p key of @p where: "where 'key'". */
        [[nodiscard]] std::string member_of(std::string_view where, std::string_view key);

        /** @brief Returns the text that names entry @p index of the list @p where. */
        [[nodiscard]] std::string entry_of(std::string_view where, std::size_t index);

        /** @brief Returns @p items, each in quotes, separated by commas. */
        [[nodiscard]] std::string list_quoted(KeySpan items);

        /**
         * @brief Reads @p member, named @p where, as a number, widening a whole number.
         * @return The number, or InvalidDescription naming @p where and the kind found.
         */
        [[nodiscard]] Result<double> read_real(const Value& member, std::string_view where);

        /**
         * @brief Reads @p member, named @p where, as a whole number.
         * @return The number, or InvalidDescription naming @p where and the kind found.
         */
        [[nodiscard]] Result<std::int64_t> read_whole(const Value& member, std::string_view where);

        /**
         * @brief Reads @p member, named @p where, as a number float32 holds as a finite value.
         * @return The float, or InvalidDescription naming @p where and the kind or value found.
         */
        [[nodiscard]] Result<float> read_float32(const Value& member, std::string_view where);

        /**
         * @brief Reads @p member, named @p where, as a string.
         * @return The string, which @p member owns, or InvalidDescription naming @p where.
         */
        [[nodiscard]] Result<const std::string*> read_string(const Value& member,
                                                             std::string_view where);

        /**
         * @brief Reads @p member, named @p where, as an array.
         * @return The entries, which @p member owns, or InvalidDescription naming @p where.
         */
        [[nodiscard]] Result<const std::vector<Value>*> read_array(const Value& member,
                                                                   std::string_view where);

        /**
         * @brief Reads the keys of one section; every refusal starts with the section's name.
         * @pre The section and the text of its name outlive the reader.
         */
        class SectionReader final
        {
            public:
                /** @brief Binds @p section and @p where, the name a refusal gives it. */
                SectionReader(const Value& section, std::string_view where) noexcept;

                /** @brief Returns InvalidDescription unless the section is an object. */
                [[nodiscard]] Status check_object() const;

                /**
                 * @brief Returns InvalidDescription for the first key outside @p allowed.
                 * @return The refusal names the key and lists @p allowed.
                 */
                [[nodiscard]] Status check_keys(KeySpan allowed) const;

                /**
                 * @brief Checks that @p key, when the section states it, holds @p expected.
                 * @param what Names the thing being read with its article, "a step schedule".
                 * @return Success for an absent key; InvalidDescription for a value that is
                 *         no string or another string.
                 */
                [[nodiscard]] Status check_type(std::string_view key, std::string_view expected,
                                                std::string_view what) const;

                /** @brief Reports whether the section states @p key. */
                [[nodiscard]] bool has(std::string_view key) const noexcept;

                /** @brief Returns the text that names the member @p key of the section. */
                [[nodiscard]] std::string member(std::string_view key) const;

                /**
                 * @brief Finds the member @p key.
                 * @pre The section is an object.
                 * @return The member, never null, or InvalidDescription naming the absent key.
                 */
                [[nodiscard]] Result<const Value*> find(std::string_view key) const;

                /**
                 * @brief Finds the member @p key and checks that it holds @p kind.
                 * @pre The section is an object.
                 * @return The member, never null, or InvalidDescription naming the key and
                 *         the kind found.
                 */
                [[nodiscard]] Result<const Value*> find_of(std::string_view key, Kind kind) const;

            private:
                /// @brief The section read.
                const Value* section_ = nullptr;

                /// @brief The name a refusal gives the section.
                std::string_view where_;
        };

    } // namespace Config
} // namespace SushiAI
