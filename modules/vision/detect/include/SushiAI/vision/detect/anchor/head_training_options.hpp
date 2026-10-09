/**************************************************************************/
/* head_training_options.hpp                                              */
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
 * @file head_training_options.hpp
 * @brief Declares the options that shape the training delta of an anchor head.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>

#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Names how the box entries of an assigned slot are pulled toward a truth. */
        enum class BoxLoss : std::uint8_t
        {
            SquaredError, /**< Each box entry moves toward its own target value. */
            CompleteIou,  /**< The box moves along the gradient of its complete IoU. */
        };

        /** @brief Names the objectness an assigned or an ignored slot is pulled toward. */
        enum class ObjectnessTarget : std::uint8_t
        {
            One,     /**< An assigned slot is pulled toward 1; an ignored slot is left alone. */
            BestIou, /**< An ignored slot is pulled toward its best IoU with a truth. */
        };

        /** @brief Holds what the delta of a head needs besides the head's decode options. */
        struct HeadTrainingOptions
        {
            AssignmentThresholds thresholds;      /**< Shape IoU to match, box IoU to ignore. */
            BoxLoss box_loss = BoxLoss::SquaredError;                   /**< Box term. */
            ObjectnessTarget objectness_target = ObjectnessTarget::One; /**< Objectness term. */
            float box_weight = 0.75f;   /**< Factor of the box deltas. */
            float object_weight = 1.0f; /**< Factor of the objectness deltas. */
            float class_weight = 1.0f;  /**< Factor of the class deltas. */
            float max_delta = 0.0f;     /**< Bound of a box delta's magnitude; 0 sets none. */
            float class_gate = 0.25f;   /**< Class probability a slot needs to be ignored. */
        };

        /**
         * @brief Checks that every training option holds a value the delta can use.
         * @return Success, or InvalidDescription for a match threshold outside (0, 1], an
         *         ignore threshold or a class gate outside [0, 1], a weight or a bound that
         *         is negative or not finite, or an enumerator outside its enumeration.
         */
        [[nodiscard]] Status validate_head_training(const HeadTrainingOptions& options);

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
