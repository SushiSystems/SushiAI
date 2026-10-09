/**************************************************************************/
/* value.hpp                                                              */
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
 * @file value.hpp
 * @brief Declares dynamically-typed configuration values and JSON parsers.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Config
    {
        /** @brief What a @ref Value holds. */
        enum class Kind : std::uint8_t
        {
            /** @brief JSON null. */
            Null,
            /** @brief true or false. */
            Bool,
            /** @brief A whole number. */
            Int,
            /** @brief A number with a fractional part or an exponent. */
            Real,
            /** @brief A string. */
            String,
            /** @brief An ordered list of values. */
            Array,
            /** @brief A set of key/value pairs, in the order the file spelled them. */
            Object,
        };

        /**
         * @brief A printable name for @p kind, for a diagnostic.
         * @param kind The kind to name.
         * @return A null-terminated literal owned by the library; never null.
         */
        std::string_view kind_name(Kind kind) noexcept;

        /**
         * @class Value
         * @brief One node of a parsed configuration document.
         */
        class Value
        {
            public:
                /** @brief Constructs the null value. */
                Value() = default;

                /** @brief Builds a boolean. @param value Its value. @return The node. */
                static Value boolean(bool value);

                /** @brief Builds a whole number. @param value Its value. @return The node. */
                static Value integer(std::int64_t value);

                /** @brief Builds a real. @param value Its value. @return The node. */
                static Value real(double value);

                /** @brief Builds a string. @param value Its value; moved in. @return The node. */
                static Value string(std::string value);

                /** @brief Builds an array. @param entries Its entries; moved in. @return The node. */
                static Value array(std::vector<Value> entries);

                /**
                 * @brief Builds an object.
                 * @param keys Its keys, in file order; moved in.
                 * @param values Its values, positionally matching @p keys; moved in.
                 * @return The node.
                 * @throws Error If the two vectors are of different lengths.
                 */
                static Value object(std::vector<std::string> keys, std::vector<Value> values);

                /** @brief What this node holds. */
                Kind kind() const noexcept { return kind_; }

                /** @brief Whether this node is JSON null. */
                bool is_null() const noexcept { return kind_ == Kind::Null; }

                /**
                 * @brief Reads this value as a boolean.
                 * @param context What this value is, for the diagnostic, e.g. "layer 3 'use_bias'".
                 * @return The boolean.
                 * @throws Error If this is not a boolean; names @p context and the kind found.
                 */
                bool as_bool(std::string_view context) const;

                /**
                 * @brief Reads this value as a whole number.
                 * @param context What this value is, for the diagnostic.
                 * @return The number.
                 * @throws Error If this is not a whole number; names @p context and the kind found.
                 */
                std::int64_t as_int(std::string_view context) const;

                /**
                 * @brief Reads this value as a real number, widening integers.
                 * @param context Context string for error diagnostics.
                 * @return Floating-point numeric value.
                 * @throws Error If value is neither real nor integer.
                 */
                double as_real(std::string_view context) const;

                /**
                 * @brief Reads this value as a string.
                 * @param context What this value is, for the diagnostic.
                 * @return The string.
                 * @throws Error If this is not a string; names @p context and the kind found.
                 */
                const std::string& as_string(std::string_view context) const;

                /**
                 * @brief Reads this value as an array.
                 * @param context What this value is, for the diagnostic.
                 * @return Its entries.
                 * @throws Error If this is not an array; names @p context and the kind found.
                 */
                const std::vector<Value>& as_array(std::string_view context) const;

                /**
                 * @brief Whether this object holds @p key; false for any non-object.
                 * @param key The key to look for.
                 * @return True when this is an object carrying @p key.
                 */
                bool has(std::string_view key) const noexcept;

                /**
                 * @brief Accesses child value associated with @p key.
                 * @param key Property key to look up.
                 * @param context Context string for error diagnostics.
                 * @return Const reference to value mapped to @p key.
                 * @throws Error If node is not an object or key is absent.
                 */
                const Value& at(std::string_view key, std::string_view context) const;

                /**
                 * @brief This object's value for @p key, or @p fallback when absent.
                 * @param key The key to look up.
                 * @param fallback Returned when @p key is absent; must outlive the result.
                 * @return The value, or @p fallback.
                 */
                const Value& at_or(std::string_view key, const Value& fallback) const noexcept;

                /** @brief This object's keys, in the order the file spelled them. */
                const std::vector<std::string>& keys() const noexcept { return keys_; }

                /**
                 * @brief Rejects any key of this object absent from @p allowed.
                 * @param context Context string for error diagnostics.
                 * @param allowed Permitted property key names.
                 * @throws Error If an unrecognized key is present.
                 */
                void reject_unknown_keys(std::string_view context,
                                         SushiRuntime::span<const std::string_view> allowed) const;

                /**
                 * @brief Returns a copy of this object with @p key removed.
                 * @param key Property key to omit.
                 * @return New value copy omitting specified key.
                 */
                Value without(std::string_view key) const;

            private:
                Kind kind_ = Kind::Null;
                bool bool_ = false;
                std::int64_t int_ = 0;
                double real_ = 0.0;
                std::string string_;
                std::vector<Value> array_;
                std::vector<std::string> keys_;
                std::vector<Value> values_;
        };

        /**
         * @brief Validates document version field against expected version.
         * @param root Document root configuration object.
         * @param key Version property key name.
         * @param expected Expected integer version identifier.
         * @param origin Origin context description for diagnostics.
         * @throws Error If version property is missing or mismatched.
         */
        void require_version(const Value& root, std::string_view key,
                             std::int64_t expected, std::string_view origin);

        /**
         * @brief Parses @p text as a JSON document.
         * @param text The document.
         * @param origin What to call it in a diagnostic, usually a path.
         * @return The parsed tree.
         * @throws Error If @p text is not valid JSON; names @p origin and the
         *         parser's own complaint.
         */
        Value parse_json(std::string_view text, std::string_view origin);

        /**
         * @brief Reads and parses the file at @p path.
         * @param path The file to read.
         * @return The parsed tree.
         * @throws Error If the file cannot be read, or is not valid JSON.
         */
        Value parse_json_file(const std::filesystem::path& path);

    } // namespace Config
} // namespace SushiAI
