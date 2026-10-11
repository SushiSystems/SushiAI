/**************************************************************************/
/* random.hpp                                                             */
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
 * @file random.hpp
 * @brief Declares the lowering rows of the operations that draw at each replay.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records dropout: the mask of the run, saved, and its product with x. */
        [[nodiscard]] Status lower_dropout(const LoweringContext& context);

        /** @brief Pairs each operation that draws at each replay with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 1> K_RANDOM_LOWERINGS = {{
            {Ops::K_DROPOUT.id, &lower_dropout},
        }};
    } // namespace Graph
} // namespace SushiAI
