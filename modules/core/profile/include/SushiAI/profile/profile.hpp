/**************************************************************************/
/* profile.hpp                                                            */
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
 * @file profile.hpp
 * @brief Declares SA_ZONE, SA_ZONE_FINE, SA_COUNTER and SA_VALUE, SushiAI's front on SushiProfiler's macros.
 * @author Mustafa Garip
 * @see modules/core/profile/README.md
 */

#pragma once

#include <SushiAI/profile/profile_config.hpp>
#include <SushiAI/profile/zone_groups.hpp>

/**
 * @brief Stops the compile when @p group is not one of SushiAI::Profile::K_ZONE_GROUPS.
 * @param group A string literal naming the zone group.
 */
#define SA_REQUIRE_ZONE_GROUP(group)                                                               \
    static_assert(::SushiAI::Profile::is_zone_group(group),                                        \
                  "the zone group is not in SushiAI::Profile::K_ZONE_GROUPS")

#if SA_ENABLE_PROFILER

#include <SushiProfiler/zones/macros.hpp>

/**
 * @brief Records the rest of the scope as the stage zone `<group>.<stage>` of group @p group.
 * @param group A string literal naming one of K_ZONE_GROUPS, such as "host".
 * @param stage A string literal naming the stage, such as "readback".
 */
#define SA_ZONE(group, stage)                                                                      \
    SA_REQUIRE_ZONE_GROUP(group);                                                                  \
    SP_ZONE_AT(SA_PROFILE_LEVEL, 1, group "." stage, group)

/** @brief Records the rest of the scope as a fine zone, kept at level 2; arguments as SA_ZONE. */
#define SA_ZONE_FINE(group, stage)                                                                 \
    SA_REQUIRE_ZONE_GROUP(group);                                                                  \
    SP_ZONE_AT(SA_PROFILE_LEVEL, 2, group "." stage, group)

/** @brief Records @p value at the counter @p name, folded per period by the CounterOp @p op. */
#define SA_COUNTER(name, unit, op, value) SP_COUNTER_AT(SA_PROFILE_LEVEL, 1, name, unit, op, value)

/** @brief Records @p value as a sample of the value site @p name. */
#define SA_VALUE(name, unit, value) SP_VALUE_AT(SA_PROFILE_LEVEL, 1, name, unit, value)

#else

/** @brief Checks @p group and records nothing: the build has no profiler. */
#define SA_ZONE(group, stage) SA_REQUIRE_ZONE_GROUP(group)

/** @brief Checks @p group and records nothing: the build has no profiler. */
#define SA_ZONE_FINE(group, stage) SA_REQUIRE_ZONE_GROUP(group)

/** @brief Expands to nothing, and does not evaluate @p value: the build has no profiler. */
#define SA_COUNTER(name, unit, op, value)

/** @brief Expands to nothing, and does not evaluate @p value: the build has no profiler. */
#define SA_VALUE(name, unit, value)

#endif
