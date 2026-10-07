/**************************************************************************/
/* top_k.hpp                                                              */
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
 * @file top_k.hpp
 * @brief Declares the selection of the k largest values of an array, in a fixed order.
 * @author Mustafa Garip
 * @see include/SushiAI/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Detect
    {
        /**
         * @brief Writes the indices of the @p k largest values, largest first.
         * @param k At least 0 and at most the number of values.
         * @param indices Receives k indices; equal values are ordered by ascending index.
         * @pre No value is NaN.
         * @see include/SushiAI/detect/README.md
         */
        void select_top_k(SushiRuntime::span<const float> values,
                          std::int64_t k,
                          std::vector<std::int64_t>& indices);

    } // namespace Detect
} // namespace SushiAI
