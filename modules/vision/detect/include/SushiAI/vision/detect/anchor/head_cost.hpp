/**************************************************************************/
/* head_cost.hpp                                                          */
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
 * @file head_cost.hpp
 * @brief Declares the reported cost of an anchor head's training delta.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the cost a head reports for one batch; no gradient is taken of it. */
        struct HeadCost
        {
            /// @brief Holds the sum of every squared delta entry over the batch size.
            double total = 0.0;
            /// @brief Holds total minus non_box, the share the report gives to the boxes.
            double box = 0.0;
            /// @brief Holds object_weight times the squared non-box entries over the batch size.
            double non_box = 0.0;
            /// @brief Holds the layer's own cost, which no batch size divides.
            double stored = 0.0;
        };

        /**
         * @brief Measures the reported cost of a head's delta and of its assignments.
         * @return The cost. `stored` is the sum of every squared entry for SquaredError; for
         *         CompleteIoU it is box_weight times the mean of 1 - IoU over the assignments,
         *         0 without one, plus object_weight times the squared non-box entries.
         * @pre The delta holds `batch * height * width * slots * (5 + C)` values for some
         *      slot count, and the batch, the grid and the class count are positive.
         */
        [[nodiscard]] HeadCost measure_head_cost(const HeadDelta& delta, const HeadShape& shape,
                                                 const HeadOptions& options,
                                                 const HeadTrainingOptions& training) noexcept;

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
