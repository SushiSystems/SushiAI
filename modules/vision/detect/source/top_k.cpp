/**************************************************************************/
/* top_k.cpp                                                              */
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
 * @file top_k.cpp
 * @brief Implements the selection of the k largest values of an array.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/top_k.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>

#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Vision::Detect
    {
        void select_top_k(SushiRuntime::span<const float> values,
                          std::int64_t k,
                          std::vector<std::int64_t>& indices)
        {
            SA_THROW_IF(k < 0 || static_cast<std::size_t>(k) > values.size(),
                        "a top-k of {} was asked of {} values", k, values.size());

            indices.resize(values.size());
            std::iota(indices.begin(), indices.end(), std::int64_t{0});

            const auto ranks_before = [values](std::int64_t left, std::int64_t right) noexcept
            {
                const float left_value = values[static_cast<std::size_t>(left)];
                const float right_value = values[static_cast<std::size_t>(right)];
                if (left_value != right_value)
                    return left_value > right_value;
                return left < right;
            };
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), ranks_before);
            indices.resize(static_cast<std::size_t>(k));
        }

    } // namespace Vision::Detect
} // namespace SushiAI
