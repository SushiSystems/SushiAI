/**************************************************************************/
/* assigner_config.hpp                                                    */
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
 * @file assigner_config.hpp
 * @brief Declares the settings of the task-aligned label assigner and the two branch presets.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiRuntime/core/span.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Free
    {
        /**
         * @brief Holds the settings of one task-aligned assigner.
         * @see modules/vision/detect/README.md
         */
        struct AssignerConfig
        {
            std::int64_t topk = 13;        /**< Candidates kept per ground-truth box. */
            std::int64_t topk2 = 13;       /**< Candidates kept after the overlap tie break. */
            std::int64_t num_classes = 80; /**< Classes, the width of a score row. */
            float alpha = 1.0f;            /**< Exponent of the class score in the metric. */
            float beta = 6.0f;             /**< Exponent of the overlap in the metric. */
            float stride_val = 16.0f;      /**< Smallest box side the candidate test uses. */
            float eps = 1e-9f;             /**< Term added to the normalisation divisor. */
        };

        /**
         * @brief Returns the settings of the one-to-many branch.
         * @param strides Strides of the model's levels; empty means 8, 16 and 32.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] AssignerConfig one_to_many_assigner_config(
            std::int64_t num_classes, SushiRuntime::span<const float> strides);

        /**
         * @brief Returns the settings of the one-to-one branch.
         * @param strides Strides of the model's levels; empty means 8, 16 and 32.
         * @see modules/vision/detect/README.md
         */
        [[nodiscard]] AssignerConfig one_to_one_assigner_config(
            std::int64_t num_classes, SushiRuntime::span<const float> strides);

    } // namespace Vision::Detect::Free
} // namespace SushiAI
