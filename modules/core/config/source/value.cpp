/**************************************************************************/
/* value.cpp                                                              */
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
 * @file value.cpp
 * @brief Implements dynamically-typed configuration values and JSON conversion.
 * @author Mustafa Garip
 */

#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

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
             * @brief The document type parsed by this translation unit.
             *
             * Uses ordered_json to preserve key ordering as spelled in source files.
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

                // Integer and floating-point types remain distinct to prevent implicit truncation.
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
                case Kind::Null:    return "null";
                case Kind::Bool:   return "a boolean";
                case Kind::Int:    return "a whole number";
                case Kind::Real:   return "a number";
                case Kind::String: return "a string";
                case Kind::Array:  return "an array";
                case Kind::Object: return "an object";
            }
            return "an unknown kind";
        }

        Value Value::boolean(bool value) noexcept
        {
            Value node;
            node.kind_ = Kind::Bool;
            node.bool_ = value;
            return node;
        }

        Value Value::integer(std::int64_t value) noexcept
        {
            Value node;
            node.kind_ = Kind::Int;
            node.int_ = value;
            return node;
        }

        Value Value::real(double value) noexcept
        {
            Value node;
            node.kind_ = Kind::Real;
            node.real_ = value;
            return node;
        }

        Value Value::string(std::string value) noexcept
        {
            Value node;
            node.kind_ = Kind::String;
            node.string_ = std::move(value);
            return node;
        }

        Value Value::array(std::vector<Value> entries) noexcept
        {
            Value node;
            node.kind_ = Kind::Array;
            node.array_ = std::move(entries);
            return node;
        }

        Value Value::object(std::vector<std::string> keys, std::vector<Value> values) noexcept
        {
            SA_FATAL_IF(keys.size() != values.size(),
                        "a config object was built from {} keys and {} values",
                        keys.size(), values.size());

            Value node;
            node.kind_ = Kind::Object;
            node.keys_ = std::move(keys);
            node.values_ = std::move(values);
            return node;
        }

        Result<bool> Value::as_bool(std::string_view context) const noexcept
        {
            SA_FAIL_IF(kind_ != Kind::Bool, StatusCode::InvalidDescription,
                       "{} must be a boolean, but it is {}", context, kind_name(kind_));
            return bool_;
        }

        Result<std::int64_t> Value::as_int(std::string_view context) const noexcept
        {
            SA_FAIL_IF(kind_ != Kind::Int, StatusCode::InvalidDescription,
                       "{} must be a whole number, but it is {}", context, kind_name(kind_));
            return int_;
        }

        Result<double> Value::as_real(std::string_view context) const noexcept
        {
            if (kind_ == Kind::Int) return static_cast<double>(int_);

            SA_FAIL_IF(kind_ != Kind::Real, StatusCode::InvalidDescription,
                       "{} must be a number, but it is {}", context, kind_name(kind_));
            return real_;
        }

        Result<const std::string&> Value::as_string(std::string_view context) const noexcept
        {
            SA_FAIL_IF(kind_ != Kind::String, StatusCode::InvalidDescription,
                       "{} must be a string, but it is {}", context, kind_name(kind_));
            return string_;
        }

        Result<const std::vector<Value>&> Value::as_array(std::string_view context) const noexcept
        {
            SA_FAIL_IF(kind_ != Kind::Array, StatusCode::InvalidDescription,
                       "{} must be an array, but it is {}", context, kind_name(kind_));
            return array_;
        }

        bool Value::has(std::string_view key) const noexcept
        {
            if (kind_ != Kind::Object) return false;

            for (const std::string& candidate : keys_)
                if (candidate == key) return true;
            return false;
        }

        Result<const Value&> Value::at(std::string_view key, std::string_view context) const noexcept
        {
            SA_FAIL_IF(kind_ != Kind::Object, StatusCode::InvalidDescription,
                       "{} must be an object carrying '{}', but it is {}",
                       context, key, kind_name(kind_));

            for (std::size_t i = 0; i < keys_.size(); ++i)
                if (keys_[i] == key) return values_[i];

            return Status::from_message(StatusCode::InvalidDescription, [&]()
            {
                return SushiLog::format_text("{} has no '{}'; it carries {}",
                                             context, key, quoted_list(keys_));
            });
        }

        const Value& Value::at_or(std::string_view key, const Value& fallback) const noexcept
        {
            if (kind_ != Kind::Object) return fallback;

            for (std::size_t i = 0; i < keys_.size(); ++i)
                if (keys_[i] == key) return values_[i];
            return fallback;
        }

        Status Value::reject_unknown_keys(std::string_view context,
                                          SushiRuntime::span<const std::string_view> allowed) const noexcept
        {
            // Non-object nodes contain no keys to validate against the allowed list.
            if (kind_ != Kind::Object) return Status();

            for (const std::string& key : keys_)
            {
                bool permitted = false;
                for (std::size_t i = 0; i < allowed.size(); ++i)
                    if (allowed[i] == key) { permitted = true; break; }

                SA_FAIL_IF(!permitted, StatusCode::InvalidDescription,
                           "{} does not accept '{}'; it accepts {}. An unknown key is "
                           "an error rather than an ignored one, because a typed key "
                           "silently falling back to a default is a wrong answer with "
                           "no symptom.",
                           context, key, quoted_list(allowed));
            }
            return Status();
        }

        Value Value::without(std::string_view key) const noexcept
        {
            if (kind_ != Kind::Object) return *this;

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

        Status require_version(const Value& root, std::string_view key,
                               std::int64_t expected, std::string_view origin) noexcept
        {
            SA_FAIL_IF(root.kind() != Kind::Object, StatusCode::InvalidFile,
                       "'{}' must be an object whose first key is '{}', but it is {}",
                       origin, key, kind_name(root.kind()));

            SA_FAIL_IF(!root.has(key), StatusCode::InvalidFile,
                       "'{}' has no '{}'. Every SushiAI configuration document leads "
                       "with its version, so that a format change is an error rather "
                       "than a misread.", origin, key);

            const std::string context = "'" + std::string(origin) + "'s '"
                                      + std::string(key) + "'";
            SA_TRY_ASSIGN(const Value& stated, root.at(key, origin));
            SA_TRY_ASSIGN(const std::int64_t version, stated.as_int(context));

            SA_FAIL_IF(version != expected, StatusCode::InvalidFile,
                       "'{}' is a {} version {} document; this build reads version {}",
                       origin, key, version, expected);
            return Status();
        }

        Result<Value> parse_json(std::string_view text, std::string_view origin) noexcept
        {
            try
            {
                const Document document = Document::parse(text.begin(), text.end());
                return convert(document);
            }
            catch (const nlohmann::json::exception& error)
            {
                return Status::from_message(StatusCode::InvalidFile, [&]()
                {
                    return SushiLog::format_text("'{}' is not valid JSON: {}", origin,
                                                 error.what());
                });
            }
        }

        Result<Value> parse_json_file(const std::filesystem::path& path) noexcept
        {
            std::ifstream in(path, std::ios::binary);
            SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                       path.string());

            const std::string text{std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>()};
            SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "failed reading '{}'", path.string());

            return parse_json(text, path.string());
        }

    } // namespace Config
} // namespace SushiAI
