/**************************************************************************/
/* name_map.hpp                                                           */
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
 * @file name_map.hpp
 * @brief Declares the name map of a weight import: renames, drops and the state key.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/import_weights.hpp>

namespace SushiAI
{
    namespace IO
    {
        /**
         * @class NameMap
         * @brief Holds how the tensor names of a weight file become the record names of a model.
         *
         * A JSON object with optional keys `state_dict`, the state dictionary's key; `rename`,
         * `{"from": ..., "to": ...}` prefix rules, the first match winning; and `drop`, names
         * with no record, matched before any rename, whole or after a '.'.
         */
        class NameMap final
        {
            public:
                /**
                 * @brief Reads the map from the JSON file at @p path.
                 * @return FileNotFound or InvalidFile for a file that cannot be read or parsed;
                 *         otherwise what from_value returns.
                 */
                [[nodiscard]] static Result<NameMap> read(
                    const std::filesystem::path& path) noexcept;

                /**
                 * @brief Reads the map from JSON text.
                 * @param origin Names the text in a message.
                 * @return What from_value returns, or InvalidFile for text that is no JSON.
                 */
                [[nodiscard]] static Result<NameMap> parse(std::string_view text,
                                                           std::string_view origin) noexcept;

                /**
                 * @brief Reads the map from a parsed document.
                 * @param origin Names the document in a message.
                 * @return InvalidDescription naming the entry at fault: a root that is no object,
                 *         a key other than the three, an empty `state_dict`, a rename that lacks
                 *         `from` or `to` or holds another key or a non-text value, an empty drop
                 *         entry.
                 */
                [[nodiscard]] static Result<NameMap> from_value(const Config::Value& root,
                                                                std::string_view origin) noexcept;

                /** @brief Returns the key of the state dictionary, or empty for none. */
                [[nodiscard]] std::string_view state_key() const noexcept;

                /**
                 * @brief Returns the rename rules in file order.
                 * @return Views of this map's text; they are valid while this map is alive and
                 *         unchanged.
                 */
                [[nodiscard]] std::vector<RenameRule> rename_rules() const;

                /** @brief Returns whether @p name, as a file states it, is on the drop list. */
                [[nodiscard]] bool is_dropped(std::string_view name) const noexcept;

            private:
                /** @brief Holds one rename rule as owned text. */
                struct Rule
                {
                    std::string from; /**< The prefix to match. */
                    std::string to;   /**< The text that replaces it. */
                };

                /** @brief Holds the state dictionary key. */
                std::string state_key_;

                /** @brief Holds the rename rules in file order. */
                std::vector<Rule> rules_;

                /** @brief Holds the drop entries in file order. */
                std::vector<std::string> drops_;
        };

    } // namespace IO
} // namespace SushiAI
