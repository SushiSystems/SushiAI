/**************************************************************************/
/* head_truth.hpp                                                         */
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
 * @file head_truth.hpp
 * @brief Declares the check of the truth rows an anchor head is trained against.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /**
         * @brief Checks that a head of @p classes classes can assign every valid truth row.
         * @param batch Images of the head; the truth must hold as many.
         * @param network Size the truth's pixel corners are divided by.
         * @return Success, or InvalidDescription naming the row and its value: arrays that do
         *         not hold their shape, a network side below 1, a class outside the head's, a
         *         centre outside [0, 1) on either axis, a width or height that is not positive.
         */
        [[nodiscard]] Status validate_head_truth(const Detect::GroundTruthBatch& truth,
                                                 std::int64_t batch, std::int64_t classes,
                                                 NetworkSize network);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
