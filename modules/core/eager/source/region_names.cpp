/**************************************************************************/
/* region_names.cpp                                                       */
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
 * @file region_names.cpp
 * @brief Implements RegionNames, the map from a region's name and direction to its id.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_names.hpp>

#include <SushiAI/failure/check.hpp>

namespace SushiAI
{
    namespace Eager
    {
        namespace
        {
            /** @brief Returns the label of a region, its name followed by its direction. */
            [[nodiscard]] std::string label(std::string_view name, Direction direction)
            {
                std::string text(name);
                text += direction == Direction::Forward ? " forward" : " backward";
                return text;
            }

        } // namespace

        RegionID RegionNames::intern(const RegionInfo& info) noexcept
        {
            std::string text = label(info.name, info.direction);
            const auto found = ids_.find(text);
            if (found != ids_.end())
                return found->second;

            const RegionID id = static_cast<RegionID>(labels_.size());
            labels_.push_back(text);
            ids_.emplace(std::move(text), id);
            return id;
        }

        RegionID RegionNames::find(std::string_view name, Direction direction) const noexcept
        {
            const auto found = ids_.find(label(name, direction));
            return found != ids_.end() ? found->second : K_NO_REGION_ID;
        }

        const std::string& RegionNames::label_of(RegionID id) const noexcept
        {
            SA_FATAL_IF(id >= labels_.size(), "Region id {} was never given; {} ids exist", id,
                        labels_.size());
            return labels_[id];
        }

    } // namespace Eager
} // namespace SushiAI
