/**************************************************************************/
/* value.hpp                                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

#pragma once

/**
 * @file value.hpp
 * @brief A small typed tree, and the only thing the config layer parses into.
 *
 * @section containment Why the JSON library is not in this file
 *
 * The parser converts its own document type into @ref Value inside
 * @c src/config/value.cpp, and the name of the library that does it appears in
 * no public header. Three things follow: the dependency links @c PRIVATE, so
 * @c cmake/SushiAIConfig.cmake.in needs no @c find_dependency and the
 * out-of-tree consumer in @c tests/package/ never sees it; and replacing the
 * parser later changes no public header at all. This is the same containment
 * @c SushiRuntime::span and @c SushiBLAS::Core::format apply to @c std::span
 * and @c std::format.
 *
 * @section representation Why this is not a std::variant
 *
 * A variant over @c std::vector<Value> needs @c Value complete at the point
 * the member is declared, and what C++17 actually guarantees is that
 * @c std::vector tolerates an incomplete element type. So each alternative is
 * its own member behind a @ref Kind tag. An object's entries are two parallel
 * vectors rather than a @c vector<pair<string, Value>>, for the same
 * completeness reason and for a second one worth having: the key order is the
 * order the file spelled them, so a diagnostic listing an object's keys reads
 * the way the file does.
 *
 * A config tree is a few dozen nodes, so the members an instance is not using
 * cost nothing worth a more delicate representation.
 *
 * @section diagnostics Every accessor takes a context
 *
 * A typed accessor that throws "expected an integer" names nothing the reader
 * can act on. Each one here takes a @p context - "layer 3 'units'",
 * "optimizer 'learning_rate'" - and puts it in the message alongside the kind
 * that was actually found. The caller knows where in the file it is; the
 * @c Value does not.
 *
 * Host lane: no SushiBLAS engine, no SYCL.
 */

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
            NUL,
            /** @brief true or false. */
            BOOL,
            /** @brief A whole number. */
            INT,
            /** @brief A number with a fractional part or an exponent. */
            REAL,
            /** @brief A string. */
            STRING,
            /** @brief An ordered list of values. */
            ARRAY,
            /** @brief A set of key/value pairs, in the order the file spelled them. */
            OBJECT,
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
                bool is_null() const noexcept { return kind_ == Kind::NUL; }

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
                 * @brief Reads this value as a real.
                 *
                 * A whole number is accepted and widened, so "learning_rate": 1
                 * means what a reader expects rather than failing on a
                 * distinction JSON does not make in the file's own syntax.
                 *
                 * @param context What this value is, for the diagnostic.
                 * @return The number.
                 * @throws Error If this is neither a real nor a whole number.
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
                 * @brief This object's value for @p key.
                 * @param key The key to look up.
                 * @param context What this object is, for the diagnostic.
                 * @return The value.
                 * @throws Error If this is not an object, or @p key is absent.
                 *         The message names @p context, @p key and every key
                 *         that IS present, so a typo is visible beside what was
                 *         meant.
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
                 * @brief Refuses any key of this object that is not in @p allowed.
                 *
                 * A typed "unitss": 256 must not train silently against a
                 * default. This is the silent-wrong-answer class the rest of
                 * the codebase is built to refuse, and it is why a config key
                 * is never merely ignored.
                 *
                 * A non-object is accepted without complaint, because "this
                 * layer carries no attributes at all" is a legitimate shape and
                 * the caller that cares states its own requirement.
                 *
                 * @param context What this object is, for the diagnostic.
                 * @param allowed Every key this object may carry.
                 * @throws Error Naming the offending key and listing @p allowed.
                 */
                void reject_unknown_keys(std::string_view context,
                                         SushiRuntime::span<const std::string_view> allowed) const;

                /**
                 * @brief A copy of this object with @p key removed.
                 *
                 * What separates a discriminator from the attributes it selects:
                 * a layer's factory is handed everything except "type", so the
                 * key that chose the factory is not also a key the factory has
                 * to allow.
                 *
                 * @param key The key to drop; its absence is not an error.
                 * @return The reduced object, or a copy of this value when it is not one.
                 */
                Value without(std::string_view key) const;

            private:
                Kind kind_ = Kind::NUL;
                bool bool_ = false;
                std::int64_t int_ = 0;
                double real_ = 0.0;
                std::string string_;
                std::vector<Value> array_;
                std::vector<std::string> keys_;
                std::vector<Value> values_;
        };

        /**
         * @brief Checks @p root's version field, which must be present and exact.
         *
         * Every configuration document this library reads leads with one, for
         * the same reason the checkpoint does: the format's next change must be
         * an error rather than a misread. It is stated once here because three
         * document kinds need it and a version check written three times is a
         * version check that will eventually disagree with itself.
         *
         * @param root The document's root object.
         * @param key The version field's name, e.g. "sushiai_model".
         * @param expected The version this build reads.
         * @param origin What to call the document in a diagnostic.
         * @throws Error If @p root is not an object, @p key is absent, it is
         *         not a whole number, or it is not @p expected.
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
