/**************************************************************************/
/* target_assignment.hpp                                                  */
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
 * @file target_assignment.hpp
 * @brief Declares the interface of a rule that assigns truths to the slots of one anchor head.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <cstdint>
#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds one truth assigned to one slot of one cell of a head. */
        struct TargetAssignment
        {
            std::int64_t image = 0;     /**< Image of the batch. */
            std::int64_t slot = 0;      /**< Index into the head's anchors. */
            std::int64_t row = 0;       /**< Grid row of the cell. */
            std::int64_t col = 0;       /**< Grid column of the cell. */
            float box_x = 0.0f;         /**< Truth centre minus the cell's column, grid units. */
            float box_y = 0.0f;         /**< Truth centre minus the cell's row, grid units. */
            float box_width = 0.0f;     /**< Truth width in grid columns. */
            float box_height = 0.0f;    /**< Truth height in grid rows. */
            std::int64_t label = 0;     /**< Class of the truth. */
            std::int64_t truth_row = 0; /**< Row of the truth within its image. */
        };

        /** @brief Holds what a rule reads to assign the truths of a batch to one head. */
        struct AssignmentInput
        {
            const AnchorLayout* layout = nullptr;     /**< Anchors and every head's mask. */
            std::int64_t head_index = 0;              /**< Head of the layout assigned to. */
            HeadShape shape;                          /**< Images and grid of the head. */
            std::int64_t classes = 0;                 /**< Class count of the head. */
            const GroundTruthBatch* truth = nullptr;  /**< Truth rows in pixel corners. */
            NetworkSize network;                      /**< Size the pixel corners divide by. */
        };

        /** @brief Assigns the truths of a batch to the slots of one anchor head. */
        class AssignmentRule
        {
            public:
                /** @brief Destroys the rule. */
                virtual ~AssignmentRule() = default;

                /**
                 * @brief Writes the assignments of @p input's head into @p result.
                 * @param result Replaced by the assignments, ordered by image, slot, row and
                 *        column; one entry per slot of a cell at most.
                 * @return Success, or the refusal of the input naming what fails.
                 * @pre @p input's layout and truth are not null.
                 */
                [[nodiscard]] virtual Status assign(
                    const AssignmentInput& input, std::vector<TargetAssignment>& result) const = 0;

            protected:
                /** @brief Constructs the base of a rule. */
                AssignmentRule() = default;

                /** @brief Copies the base of a rule. */
                AssignmentRule(const AssignmentRule&) = default;

                /** @brief Moves the base of a rule. */
                AssignmentRule(AssignmentRule&&) = default;

                /** @brief Copy-assigns the base of a rule. */
                AssignmentRule& operator=(const AssignmentRule&) = default;

                /** @brief Move-assigns the base of a rule. */
                AssignmentRule& operator=(AssignmentRule&&) = default;
        };

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
