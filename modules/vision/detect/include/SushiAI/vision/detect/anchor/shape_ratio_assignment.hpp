/**************************************************************************/
/* shape_ratio_assignment.hpp                                             */
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
 * @file shape_ratio_assignment.hpp
 * @brief Declares the assignment of truths to every anchor within a side ratio, with neighbours.
 * @author Mustafa Garip
 * @see modules/vision/detect/README.md
 */

#pragma once

#include <vector>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>

namespace SushiAI
{
    namespace Vision::Detect::Anchor
    {
        /** @brief Holds the default limit on the side ratio of a truth and an anchor. */
        inline constexpr float K_DEFAULT_RATIO_LIMIT = 4.0f;

        /** @brief Holds the default offset that admits a neighbour cell. */
        inline constexpr float K_DEFAULT_NEIGHBOUR_OFFSET = 0.5f;

        /** @brief Holds the two settings of the shape-ratio assignment. */
        struct ShapeRatioOptions
        {
            /// @brief Holds the bound the larger of a side ratio and its inverse stays below.
            float ratio_limit = K_DEFAULT_RATIO_LIMIT;
            /// @brief Holds the distance from a cell edge within which the neighbour is taken.
            float neighbour_offset = K_DEFAULT_NEIGHBOUR_OFFSET;
        };

        /**
         * @brief Checks the settings of a shape-ratio assignment.
         * @return Success, or InvalidDescription naming the key: ratio_limit must be finite and
         *         above 1, neighbour_offset finite and in [0, 0.5].
         */
        [[nodiscard]] Status validate_shape_ratio_options(const ShapeRatioOptions& options);

        /**
         * @brief Assigns each truth to every anchor of a head whose side ratio stays within a
         *        limit, in the truth's cell and in the neighbours its centre lies near.
         * @see modules/vision/detect/README.md
         */
        class ShapeRatioAssignment final : public AssignmentRule
        {
            public:
                /**
                 * @brief Constructs the rule with @p options.
                 * @pre validate_shape_ratio_options(@p options) succeeds.
                 */
                explicit ShapeRatioAssignment(const ShapeRatioOptions& options) noexcept;

                /**
                 * @brief Writes the assignments of @p input's head into @p result.
                 * @return Success; the failure of validate_head_truth; or InvalidDescription for
                 *         a head index or grid the layout cannot hold.
                 */
                [[nodiscard]] Status assign(const AssignmentInput& input,
                                            std::vector<TargetAssignment>& result) const override;

            private:
                /// @brief Holds the settings the rule was built with.
                ShapeRatioOptions options_;
        };

    } // namespace Vision::Detect::Anchor
} // namespace SushiAI
