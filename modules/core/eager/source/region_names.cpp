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
 * @file region_names.cpp
 * @brief Implements RegionNames, the map from a region's name and direction to its id.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#include <SushiAI/eager/region_names.hpp>

#include <SushiAI/core/common.hpp>

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

        RegionID RegionNames::intern(const RegionInfo& info)
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

        RegionID RegionNames::find(std::string_view name, Direction direction) const
        {
            const auto found = ids_.find(label(name, direction));
            return found != ids_.end() ? found->second : K_NO_REGION_ID;
        }

        const std::string& RegionNames::label_of(RegionID id) const
        {
            SA_THROW_IF(id >= labels_.size(), "Region id {} was never given; {} ids exist", id,
                        labels_.size());
            return labels_[id];
        }

    } // namespace Eager
} // namespace SushiAI
