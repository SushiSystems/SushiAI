/**************************************************************************/
/* name_map.cpp                                                           */
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
 * @file name_map.cpp
 * @brief Defines the reading of a name map from its JSON document and its drop and rename queries.
 * @author Mustafa Garip
 */

#include <SushiAI/io/name_map.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/import_weights.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Names the key of the state dictionary. */
            constexpr std::string_view K_STATE_KEY = "state_dict";

            /** @brief Names the key of the rename rules. */
            constexpr std::string_view K_RENAME_KEY = "rename";

            /** @brief Names the key of the drop list. */
            constexpr std::string_view K_DROP_KEY = "drop";

            /** @brief Names the key of the prefix a rename rule matches. */
            constexpr std::string_view K_RULE_FROM = "from";

            /** @brief Names the key of the text a rename rule writes. */
            constexpr std::string_view K_RULE_TO = "to";

            /** @brief Lists the keys the map may hold. */
            constexpr std::string_view K_MAP_KEYS[] = {K_STATE_KEY, K_RENAME_KEY, K_DROP_KEY};

            /** @brief Lists the keys a rename rule may hold. */
            constexpr std::string_view K_RULE_KEYS[] = {K_RULE_FROM, K_RULE_TO};

            /** @brief Returns whether @p name ends with @p suffix. */
            bool ends_with(std::string_view name, std::string_view suffix) noexcept
            {
                return name.size() >= suffix.size() &&
                       name.substr(name.size() - suffix.size()) == suffix;
            }

            /** @brief Returns the text "<what> <index> of <origin>" a message quotes. */
            std::string entry_context(std::string_view what, std::size_t index,
                                      std::string_view origin)
            {
                return std::string(what) + " " + std::to_string(index) + " of '" +
                       std::string(origin) + "'";
            }
        } // namespace

        Result<NameMap> NameMap::read(const std::filesystem::path& path) noexcept
        {
            SA_TRY_ASSIGN(const Config::Value root, Config::parse_json_file(path));
            return from_value(root, path.string());
        }

        Result<NameMap> NameMap::parse(std::string_view text, std::string_view origin) noexcept
        {
            SA_TRY_ASSIGN(const Config::Value root, Config::parse_json(text, origin));
            return from_value(root, origin);
        }

        Result<NameMap> NameMap::from_value(const Config::Value& root,
                                            std::string_view origin) noexcept
        {
            const std::string document = "the name map '" + std::string(origin) + "'";
            SA_FAIL_IF(root.kind() != Config::Kind::Object, StatusCode::InvalidDescription,
                       "{} is {}, not an object", document, Config::kind_name(root.kind()));
            SA_TRY(root.reject_unknown_keys(
                document, SushiRuntime::span<const std::string_view>(K_MAP_KEYS, 3)));

            NameMap map;
            if (root.has(K_STATE_KEY))
            {
                SA_TRY_ASSIGN(const Config::Value& stated, root.at(K_STATE_KEY, document));
                SA_TRY_ASSIGN(const std::string& key, stated.as_string("the 'state_dict' key"));
                SA_FAIL_IF(key.empty(), StatusCode::InvalidDescription,
                           "{}: 'state_dict' is empty; leave it out for a file that is the "
                           "dictionary",
                           document);
                map.state_key_ = key;
            }

            if (root.has(K_RENAME_KEY))
            {
                SA_TRY_ASSIGN(const Config::Value& stated, root.at(K_RENAME_KEY, document));
                SA_TRY_ASSIGN(const std::vector<Config::Value>& rows,
                              stated.as_array("the 'rename' list"));
                for (std::size_t i = 0; i < rows.size(); ++i)
                {
                    const std::string context = entry_context("rename rule", i, origin);
                    SA_FAIL_IF(rows[i].kind() != Config::Kind::Object,
                               StatusCode::InvalidDescription, "{} is {}, not an object", context,
                               Config::kind_name(rows[i].kind()));
                    SA_TRY(rows[i].reject_unknown_keys(
                        context, SushiRuntime::span<const std::string_view>(K_RULE_KEYS, 2)));
                    SA_TRY_ASSIGN(const Config::Value& from, rows[i].at(K_RULE_FROM, context));
                    SA_TRY_ASSIGN(const Config::Value& to, rows[i].at(K_RULE_TO, context));
                    SA_TRY_ASSIGN(const std::string& from_text,
                                  from.as_string(context + " 'from'"));
                    SA_TRY_ASSIGN(const std::string& to_text, to.as_string(context + " 'to'"));
                    map.rules_.push_back(Rule{from_text, to_text});
                }
            }

            if (root.has(K_DROP_KEY))
            {
                SA_TRY_ASSIGN(const Config::Value& stated, root.at(K_DROP_KEY, document));
                SA_TRY_ASSIGN(const std::vector<Config::Value>& rows,
                              stated.as_array("the 'drop' list"));
                for (std::size_t i = 0; i < rows.size(); ++i)
                {
                    const std::string context = entry_context("drop entry", i, origin);
                    SA_TRY_ASSIGN(const std::string& name, rows[i].as_string(context));
                    SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription, "{} is empty",
                               context);
                    map.drops_.push_back(name);
                }
            }

            return map;
        }

        std::string_view NameMap::state_key() const noexcept
        {
            return state_key_;
        }

        std::vector<RenameRule> NameMap::rename_rules() const
        {
            std::vector<RenameRule> rules;
            rules.reserve(rules_.size());
            for (const Rule& rule : rules_)
                rules.push_back(RenameRule{rule.from, rule.to});
            return rules;
        }

        const std::vector<std::string>& NameMap::drop_names() const noexcept
        {
            return drops_;
        }

        bool NameMap::is_dropped(std::string_view name) const noexcept
        {
            for (const std::string& drop : drops_)
            {
                if (name == drop)
                    return true;
                if (name.size() > drop.size() && ends_with(name, drop) &&
                    name[name.size() - drop.size() - 1] == '.')
                    return true;
            }
            return false;
        }

    } // namespace IO
} // namespace SushiAI
