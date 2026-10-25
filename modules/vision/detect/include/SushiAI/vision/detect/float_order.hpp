/**************************************************************************/
/* float_order.hpp                                                        */
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
 * @file float_order.hpp
 * @brief Declares the finiteness scan and the descending order that tolerate NaN.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        /**
         * @brief Returns the index of the first NaN or infinity in @p values, or -1 if none.
         */
        inline std::int64_t first_non_finite(SushiRuntime::span<const float> values) noexcept
        {
            for (std::size_t at = 0; at < values.size(); ++at)
            {
                if (!std::isfinite(values[at]))
                    return static_cast<std::int64_t>(at);
            }
            return -1;
        }

        /**
         * @brief Returns whether @p left ranks strictly before @p right, largest value first.
         * @return A NaN ranks below every number and equal to another NaN, so the relation is a
         *         strict weak ordering whatever the data holds.
         */
        inline bool ranks_above(float left, float right) noexcept
        {
            if (std::isnan(left))
                return false;
            if (std::isnan(right))
                return true;
            return left > right;
        }

    } // namespace Vision::Detect
} // namespace SushiAI
