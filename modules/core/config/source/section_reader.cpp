/**************************************************************************/
/* section_reader.cpp                                                     */
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
 * @file section_reader.cpp
 * @brief Implements the readers of one object of a configuration document that return a status.
 * @author Mustafa Garip
 */

#include <SushiAI/config/section_reader.hpp>

#include <cmath>
#include <limits>

namespace SushiAI
{
    namespace Config
    {
        std::string member_of(std::string_view where, std::string_view key)
        {
            std::string text(where);
            text += " '";
            text.append(key.data(), key.size());
            text += "'";
            return text;
        }

        std::string entry_of(std::string_view where, std::size_t index)
        {
            return std::string(where) + " entry " + std::to_string(index);
        }

        std::string list_quoted(KeySpan items)
        {
            std::string text;
            for (const std::string_view item : items)
            {
                text += text.empty() ? "'" : ", '";
                text.append(item.data(), item.size());
                text += "'";
            }
            return text;
        }

        Result<double> read_real(const Value& member, std::string_view where)
        {
            return member.as_real(where);
        }

        Result<std::int64_t> read_whole(const Value& member, std::string_view where)
        {
            return member.as_int(where);
        }

        Result<float> read_float32(const Value& member, std::string_view where)
        {
            SA_TRY_ASSIGN(const double wide, read_real(member, where));

            const double limit = static_cast<double>(std::numeric_limits<float>::max());
            SA_FAIL_IF(!std::isfinite(wide) || std::fabs(wide) > limit,
                       StatusCode::InvalidDescription,
                       "{} is {}, which float32 does not hold as a finite value", where, wide);
            return static_cast<float>(wide);
        }

        Result<const std::string*> read_string(const Value& member, std::string_view where)
        {
            SA_TRY_ASSIGN(const std::string& text, member.as_string(where));
            return &text;
        }

        Result<const std::vector<Value>*> read_array(const Value& member, std::string_view where)
        {
            SA_TRY_ASSIGN(const std::vector<Value>& entries, member.as_array(where));
            return &entries;
        }

        SectionReader::SectionReader(const Value& section, std::string_view where) noexcept
            : section_(&section), where_(where)
        {

        }

        Status SectionReader::check_object() const
        {
            SA_FAIL_IF(section_->kind() != Kind::Object, StatusCode::InvalidDescription,
                       "{} must be an object, but it is {}", where_,
                       kind_name(section_->kind()));
            return Status();
        }

        Status SectionReader::check_keys(KeySpan allowed) const
        {
            for (const std::string& key : section_->keys())
            {
                bool known = false;
                for (const std::string_view candidate : allowed)
                    known = known || candidate == key;

                SA_FAIL_IF(!known, StatusCode::InvalidDescription,
                           "{} states the key '{}', which it does not have; it accepts {}",
                           where_, key, list_quoted(allowed));
            }
            return Status();
        }

        Status SectionReader::check_type(std::string_view key, std::string_view expected,
                                         std::string_view what) const
        {
            if (!section_->has(key))
                return Status();

            const std::string where = member(key);
            SA_TRY_ASSIGN(const Value& member_value, section_->at(key, where));
            SA_TRY_ASSIGN(const std::string* const stated, read_string(member_value, where));
            SA_FAIL_IF(*stated != expected, StatusCode::InvalidDescription,
                       "{} is '{}'; {} states '{}'", where, *stated, what, expected);
            return Status();
        }

        bool SectionReader::has(std::string_view key) const noexcept
        {
            return section_->has(key);
        }

        std::string SectionReader::member(std::string_view key) const
        {
            return member_of(where_, key);
        }

        Result<const Value*> SectionReader::find(std::string_view key) const
        {
            SA_FAIL_IF(!section_->has(key), StatusCode::InvalidDescription, "{} has no '{}'",
                       where_, key);
            SA_TRY_ASSIGN(const Value& found, section_->at(key, where_));
            return &found;
        }

        Result<const Value*> SectionReader::find_of(std::string_view key, Kind kind) const
        {
            SA_TRY_ASSIGN(const Value* const found, find(key));
            SA_FAIL_IF(found->kind() != kind, StatusCode::InvalidDescription,
                       "{} must be {}, but it is {}", member(key), kind_name(kind),
                       kind_name(found->kind()));
            return found;
        }

    } // namespace Config
} // namespace SushiAI
