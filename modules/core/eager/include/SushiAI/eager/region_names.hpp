/**************************************************************************/
/* region_names.hpp                                                       */
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
 * @file region_names.hpp
 * @brief Declares RegionNames, which gives each region one id for the life of a session.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <SushiAI/eager/partition.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Identifies a region across steps, whatever key a step gives it. */
        using RegionID = std::uint32_t;

        /** @brief Stands for "no region has this name". */
        inline constexpr RegionID K_NO_REGION_ID = ~RegionID{0};

        /**
         * @brief Maps a region's name and direction to one id, the same on every step.
         * @see modules/core/eager/README.md
         */
        class RegionNames final
        {
            public:
                /** @brief Returns the id of @p info's name and direction, giving one if new. */
                [[nodiscard]] RegionID intern(const RegionInfo& info) noexcept;

                /** @brief Returns the id of @p name and @p direction, or K_NO_REGION_ID. */
                [[nodiscard]] RegionID find(std::string_view name, Direction direction) const noexcept;

                /**
                 * @brief Returns the label of @p id, such as "head#0 backward".
                 * @pre @p id came from intern() or find() of this object.
                 */
                [[nodiscard]] const std::string& label_of(RegionID id) const noexcept;

                /** @brief Returns how many ids have been given. */
                [[nodiscard]] std::size_t size() const noexcept
                {
                    return labels_.size();
                }

            private:
                /// @brief The id of each label.
                std::unordered_map<std::string, RegionID> ids_;

                /// @brief The label of each id, by id.
                std::vector<std::string> labels_;
        };

    } // namespace Eager
} // namespace SushiAI
