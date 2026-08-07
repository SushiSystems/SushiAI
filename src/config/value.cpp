/**************************************************************************/
/* value.cpp                                                              */
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

// THE ONLY FILE IN THIS REPOSITORY THAT NAMES nlohmann::json.
//
// That is the whole reason Config::Value exists: the dependency links PRIVATE,
// SushiAIConfig.cmake.in needs no find_dependency, and the out-of-tree
// consumer in tests/package/ never sees it. Adding an #include of it anywhere
// else silently gives that away - the build would still succeed, and the
// breakage would surface in somebody else's project.

#include <SushiAI/config/value.hpp>

#include <fstream>
#include <iterator>
#include <sstream>
#include <utility>
#include <nlohmann/json.hpp>

namespace SushiAI
{
    namespace Config
    {
        namespace
        {
            /**
             * @brief The document type this file parses into, before conversion.
             *
             * @c ordered_json rather than @c json, because the default keeps an
             * object's members in a @c std::map and therefore hands them back
             * in alphabetical order. Config::Value promises the order the file
             * spelled - a "this object carries 'a', 'b', 'c'" diagnostic should
             * read the way the file reads - and that promise has to be kept
             * here, since by the time a Value exists the order is already gone.
             */
            using Document = nlohmann::ordered_json;

            /**
             * @brief Converts one parsed node into a Value, recursively.
             * @param node The parsed node.
             * @return The equivalent Value.
             */
            Value convert(const Document& node)
            {
                if (node.is_null())    return Value{};
                if (node.is_boolean()) return Value::boolean(node.get<bool>());

                // Integer and real are kept apart, so as_int can refuse 2.5
                // rather than truncating it. JSON's own grammar makes the
                // distinction; erasing it here would be this layer's choice and
                // a silent one.
                if (node.is_number_integer() || node.is_number_unsigned())
                    return Value::integer(node.get<std::int64_t>());
                if (node.is_number_float()) return Value::real(node.get<double>());

                if (node.is_string()) return Value::string(node.get<std::string>());

                if (node.is_array())
                {
                    std::vector<Value> entries;
                    entries.reserve(node.size());
                    for (const Document& entry : node)
                        entries.push_back(convert(entry));
                    return Value::array(std::move(entries));
                }

                std::vector<std::string> keys;
                std::vector<Value> values;
                keys.reserve(node.size());
                values.reserve(node.size());
                for (auto it = node.begin(); it != node.end(); ++it)
                {
                    keys.push_back(it.key());
                    values.push_back(convert(it.value()));
                }
                return Value::object(std::move(keys), std::move(values));
            }

            /** @brief Renders @p keys as "'a', 'b', 'c'", for a diagnostic. */
            std::string quoted_list(const std::vector<std::string>& keys)
            {
                if (keys.empty()) return "(none)";

                std::string out;
                for (const std::string& key : keys)
                {
                    if (!out.empty()) out += ", ";
                    out += "'" + key + "'";
                }
                return out;
            }

            /** @brief Renders @p names as "'a', 'b', 'c'", for a diagnostic. */
            std::string quoted_list(SushiRuntime::span<const std::string_view> names)
            {
                if (names.size() == 0) return "(none)";

                std::string out;
                for (std::size_t i = 0; i < names.size(); ++i)
                {
                    if (i != 0) out += ", ";
                    out += "'";
                    out.append(names[i].data(), names[i].size());
                    out += "'";
                }
                return out;
            }
        }

        std::string_view kind_name(Kind kind) noexcept
        {
            switch (kind)
            {
                case Kind::NUL:    return "null";
                case Kind::BOOL:   return "a boolean";
                case Kind::INT:    return "a whole number";
                case Kind::REAL:   return "a number";
                case Kind::STRING: return "a string";
                case Kind::ARRAY:  return "an array";
                case Kind::OBJECT: return "an object";
            }
            return "an unknown kind";
        }

        // --- Construction ----------------------------------------------------

        Value Value::boolean(bool value)
        {
            Value node;
            node.kind_ = Kind::BOOL;
            node.bool_ = value;
            return node;
        }

        Value Value::integer(std::int64_t value)
        {
            Value node;
            node.kind_ = Kind::INT;
            node.int_ = value;
            return node;
        }

        Value Value::real(double value)
        {
            Value node;
            node.kind_ = Kind::REAL;
            node.real_ = value;
            return node;
        }

        Value Value::string(std::string value)
        {
            Value node;
            node.kind_ = Kind::STRING;
            node.string_ = std::move(value);
            return node;
        }

        Value Value::array(std::vector<Value> entries)
        {
            Value node;
            node.kind_ = Kind::ARRAY;
            node.array_ = std::move(entries);
            return node;
        }

        Value Value::object(std::vector<std::string> keys, std::vector<Value> values)
        {
            SA_THROW_IF(keys.size() != values.size(),
                        "a config object was built from {} keys and {} values",
                        keys.size(), values.size());

            Value node;
            node.kind_ = Kind::OBJECT;
            node.keys_ = std::move(keys);
            node.values_ = std::move(values);
            return node;
        }

        // --- Typed access ----------------------------------------------------

        bool Value::as_bool(std::string_view context) const
        {
            SA_THROW_IF(kind_ != Kind::BOOL,
                        "{} must be a boolean, but it is {}", context, kind_name(kind_));
            return bool_;
        }

        std::int64_t Value::as_int(std::string_view context) const
        {
            SA_THROW_IF(kind_ != Kind::INT,
                        "{} must be a whole number, but it is {}", context, kind_name(kind_));
            return int_;
        }

        double Value::as_real(std::string_view context) const
        {
            if (kind_ == Kind::INT) return static_cast<double>(int_);

            SA_THROW_IF(kind_ != Kind::REAL,
                        "{} must be a number, but it is {}", context, kind_name(kind_));
            return real_;
        }

        const std::string& Value::as_string(std::string_view context) const
        {
            SA_THROW_IF(kind_ != Kind::STRING,
                        "{} must be a string, but it is {}", context, kind_name(kind_));
            return string_;
        }

        const std::vector<Value>& Value::as_array(std::string_view context) const
        {
            SA_THROW_IF(kind_ != Kind::ARRAY,
                        "{} must be an array, but it is {}", context, kind_name(kind_));
            return array_;
        }

        // --- Object access ---------------------------------------------------

        bool Value::has(std::string_view key) const noexcept
        {
            if (kind_ != Kind::OBJECT) return false;

            for (const std::string& candidate : keys_)
                if (candidate == key) return true;
            return false;
        }

        const Value& Value::at(std::string_view key, std::string_view context) const
        {
            SA_THROW_IF(kind_ != Kind::OBJECT,
                        "{} must be an object carrying '{}', but it is {}",
                        context, key, kind_name(kind_));

            for (std::size_t i = 0; i < keys_.size(); ++i)
                if (keys_[i] == key) return values_[i];

            SA_THROW_IF(true, "{} has no '{}'; it carries {}",
                        context, key, quoted_list(keys_));
            return values_[0];   // unreachable: SA_THROW_IF(true) always throws
        }

        const Value& Value::at_or(std::string_view key, const Value& fallback) const noexcept
        {
            if (kind_ != Kind::OBJECT) return fallback;

            for (std::size_t i = 0; i < keys_.size(); ++i)
                if (keys_[i] == key) return values_[i];
            return fallback;
        }

        void Value::reject_unknown_keys(std::string_view context,
                                        SushiRuntime::span<const std::string_view> allowed) const
        {
            // A non-object carries no keys to reject. "This layer has no
            // attributes at all" is a legitimate shape, and a caller that needs
            // an object says so itself.
            if (kind_ != Kind::OBJECT) return;

            for (const std::string& key : keys_)
            {
                bool permitted = false;
                for (std::size_t i = 0; i < allowed.size(); ++i)
                    if (allowed[i] == key) { permitted = true; break; }

                SA_THROW_IF(!permitted,
                            "{} does not accept '{}'; it accepts {}. An unknown key is "
                            "an error rather than an ignored one, because a typed key "
                            "silently falling back to a default is a wrong answer with "
                            "no symptom.",
                            context, key, quoted_list(allowed));
            }
        }

        Value Value::without(std::string_view key) const
        {
            if (kind_ != Kind::OBJECT) return *this;

            std::vector<std::string> keys;
            std::vector<Value> values;
            keys.reserve(keys_.size());
            values.reserve(values_.size());

            for (std::size_t i = 0; i < keys_.size(); ++i)
            {
                if (keys_[i] == key) continue;
                keys.push_back(keys_[i]);
                values.push_back(values_[i]);
            }
            return Value::object(std::move(keys), std::move(values));
        }

        // --- Parsing ---------------------------------------------------------

        Value parse_json(std::string_view text, std::string_view origin)
        {
            try
            {
                const Document document = Document::parse(text.begin(), text.end());
                return convert(document);
            }
            catch (const nlohmann::json::exception& error)
            {
                SA_THROW_IF(true, "'{}' is not valid JSON: {}", origin, error.what());
            }
            return Value{};   // unreachable: SA_THROW_IF(true) always throws
        }

        Value parse_json_file(const std::filesystem::path& path)
        {
            std::ifstream in(path, std::ios::binary);
            SA_THROW_IF(!in, "could not open '{}' for reading", path.string());

            const std::string text{std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>()};
            SA_THROW_IF(in.bad(), "failed reading '{}'", path.string());

            return parse_json(text, path.string());
        }

    } // namespace Config
} // namespace SushiAI
