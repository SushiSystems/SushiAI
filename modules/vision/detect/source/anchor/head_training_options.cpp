/**************************************************************************/
/* head_training_options.cpp                                              */
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
 * @file head_training_options.cpp
 * @brief Implements the check of an anchor head's training options.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#include <SushiAI/vision/detect/anchor/head_training_options.hpp>

#include <cmath>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        namespace
        {
            /** @brief Reports whether @p value is finite and lies in [0, 1]. */
            bool is_unit_fraction(float value) noexcept
            {
                return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
            }

            /** @brief Reports whether @p value is finite and not negative. */
            bool is_finite_magnitude(float value) noexcept
            {
                return std::isfinite(value) && value >= 0.0f;
            }
        } // namespace

        Status validate_head_training(const HeadTrainingOptions& options)
        {
            const float match = options.thresholds.match_threshold;
            SA_FAIL_IF(!is_unit_fraction(match) || !(match > 0.0f),
                       StatusCode::InvalidDescription,
                       "the match threshold of an anchor head must lie in (0, 1]; got {}", match);
            SA_FAIL_IF(!is_unit_fraction(options.thresholds.ignore_threshold),
                       StatusCode::InvalidDescription,
                       "the ignore threshold of an anchor head must lie in [0, 1]; got {}",
                       options.thresholds.ignore_threshold);
            SA_FAIL_IF(!is_unit_fraction(options.class_gate), StatusCode::InvalidDescription,
                       "the class gate of an anchor head must lie in [0, 1]; got {}",
                       options.class_gate);

            SA_FAIL_IF(!is_finite_magnitude(options.box_weight), StatusCode::InvalidDescription,
                       "the box weight of an anchor head must be finite and not negative; got {}",
                       options.box_weight);
            SA_FAIL_IF(!is_finite_magnitude(options.object_weight),
                       StatusCode::InvalidDescription,
                       "the object weight of an anchor head must be finite and not negative; "
                       "got {}",
                       options.object_weight);
            SA_FAIL_IF(!is_finite_magnitude(options.class_weight), StatusCode::InvalidDescription,
                       "the class weight of an anchor head must be finite and not negative; "
                       "got {}",
                       options.class_weight);
            SA_FAIL_IF(!is_finite_magnitude(options.max_delta), StatusCode::InvalidDescription,
                       "max_delta of an anchor head must be finite and not negative; got {}",
                       options.max_delta);

            SA_FAIL_IF(options.box_loss != BoxLoss::SquaredError &&
                           options.box_loss != BoxLoss::CompleteIou,
                       StatusCode::InvalidDescription,
                       "the box loss of an anchor head is outside BoxLoss; got {}",
                       static_cast<int>(options.box_loss));
            SA_FAIL_IF(options.objectness_target != ObjectnessTarget::One &&
                           options.objectness_target != ObjectnessTarget::BestIou,
                       StatusCode::InvalidDescription,
                       "the objectness target of an anchor head is outside ObjectnessTarget; "
                       "got {}",
                       static_cast<int>(options.objectness_target));
            return Status();
        }

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
