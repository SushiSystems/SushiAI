/**************************************************************************/
/* shape.hpp                                                              */
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
 * @file shape.hpp
 * @brief Declares the lowering rows of the shape operations and their table.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/graph/lowering/rule.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Checks that every result of a leaf has a tensor bound; records no task. */
        [[nodiscard]] Status lower_leaf(const LoweringContext& context) noexcept;

        /**
         * @brief Records a copy of a contiguous operand into a result of other extents, and
         *        nothing when the plan made the result a view of the operand.
         */
        [[nodiscard]] Status lower_reshape(const LoweringContext& context) noexcept;

        /** @brief Records the conversion of one operand between the NCHW and NHWC layouts. */
        [[nodiscard]] Status lower_layout_convert(const LoweringContext& context) noexcept;

        /** @brief Records a copy of the operand into the result. */
        [[nodiscard]] Status lower_stop_gradient(const LoweringContext& context) noexcept;

        /** @brief Pairs each shape operation with the row that lowers it. */
        inline constexpr std::array<LoweringEntry, 4> K_SHAPE_LOWERINGS = {{
            {Ops::K_LEAF.id, &lower_leaf},
            {Ops::K_RESHAPE.id, &lower_reshape},
            {Ops::K_LAYOUT_CONVERT.id, &lower_layout_convert},
            {Ops::K_STOP_GRADIENT.id, &lower_stop_gradient},
        }};
    } // namespace Graph
} // namespace SushiAI
