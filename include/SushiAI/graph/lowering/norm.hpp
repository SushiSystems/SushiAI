/**************************************************************************/
/* norm.hpp                                                               */
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
 * @file norm.hpp
 * @brief Declares the lowering rows of the normalization operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Records 2D batch normalization and the three batch statistics it saves. */
        [[nodiscard]] Status lower_batchnorm2d(const LoweringContext& context);

        /** @brief Records the gradients of 2D batch normalization for x, gamma and beta. */
        [[nodiscard]] Status lower_batchnorm2d_backward(const LoweringContext& context);

        /** @brief Records a per-channel scale and shift of an NHWC operand. */
        [[nodiscard]] Status lower_channel_affine(const LoweringContext& context);

        /** @brief Records the fold of saved statistics into one per-channel scale and shift. */
        [[nodiscard]] Status lower_batchnorm_fold(const LoweringContext& context);

        /** @brief Pairs each normalization operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 4> K_NORM_LOWERINGS = {{
            {Ops::K_BATCHNORM2D.id, &lower_batchnorm2d},
            {Ops::K_BATCHNORM2D_BACKWARD.id, &lower_batchnorm2d_backward},
            {Ops::K_CHANNEL_AFFINE.id, &lower_channel_affine},
            {Ops::K_BATCHNORM_FOLD.id, &lower_batchnorm_fold},
        }};
    } // namespace Graph
} // namespace SushiAI
