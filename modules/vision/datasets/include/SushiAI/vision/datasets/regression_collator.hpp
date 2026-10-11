/**************************************************************************/
/* regression_collator.hpp                                                */
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
 * @file regression_collator.hpp
 * @brief Declares the collator of the regression task over a tabular list.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <SushiAI/core/result.hpp>
#include <SushiAI/vision/datasets/collator.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the regression task in a training file. */
    inline constexpr std::string_view K_TASK_REGRESSION = "regression";

    /**
     * @brief Writes inputs [N, F] float32 and the role values: [N, K] float32.
     *
     * Row n of the inputs holds FittedSample::features of sample n, and row n of values
     * its FittedSample::targets. No image is read.
     */
    class RegressionCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator of F = @p feature_count and K = @p target_count.
             * @return The collator, never null; InvalidDescription for a count below 1.
             */
            [[nodiscard]] static Result<std::unique_ptr<RegressionCollator>> make(
                std::int32_t feature_count, std::int32_t target_count);

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /**
             * @brief Copies each sample's features into the inputs and its targets into values.
             * @return Incompatible for inputs that are not float32 of [N, F] and for a batch
             *         of another layout; InvalidDescription, opening with the file, for a
             *         sample that does not hold F features and K targets.
             */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the two counts and the description of values. */
            RegressionCollator(std::int32_t feature_count, std::int32_t target_count);

            /// @brief Counts the features of one sample, F.
            std::int32_t feature_count_;

            /// @brief Counts the targets of one sample, K.
            std::int32_t target_count_;

            /// @brief Holds the description of the values role.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Vision::Datasets
