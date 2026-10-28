/**************************************************************************/
/* zone_groups.hpp                                                        */
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
 * @file zone_groups.hpp
 * @brief Declares K_ZONE_GROUPS and is_zone_group, the one list of the zone groups SushiAI's sites use.
 * @author Mustafa Garip
 * @see modules/core/profile/README.md
 */

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace SushiAI
{
    namespace Profile
    {
        /** @brief Holds the number of zone groups SushiAI's own sites use. */
        inline constexpr std::size_t K_ZONE_GROUP_COUNT = 10U;

        /**
         * @brief Holds the zone groups SushiAI's own sites use, as SA_ZONE names them.
         *
         * SA_ZONE refuses a group that is not in this list when it compiles. The groups of the
         * libraries below SushiAI, such as "Runtime" and "BLAS", are not in it.
         */
        inline constexpr std::array<std::string_view, K_ZONE_GROUP_COUNT> K_ZONE_GROUPS = {
            "compile", "copy", "data", "device", "eager", "eval", "host", "io", "optim", "train"};

        /** @brief Returns whether @p group is one of K_ZONE_GROUPS. */
        [[nodiscard]] constexpr bool is_zone_group(std::string_view group) noexcept
        {
            for (const std::string_view known : K_ZONE_GROUPS)
            {
                if (known == group)
                    return true;
            }
            return false;
        }
    } // namespace Profile
} // namespace SushiAI
