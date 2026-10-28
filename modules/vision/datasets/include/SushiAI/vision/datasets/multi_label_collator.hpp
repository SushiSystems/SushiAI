/**************************************************************************/
/* multi_label_collator.hpp                                               */
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
 * @file multi_label_collator.hpp
 * @brief Declares the collator of the multi-label task.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/vision/datasets/collator.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /** @brief Names the multi-label task in a training file. */
    inline constexpr std::string_view K_TASK_MULTI_LABEL = "multi_label";

    /**
     * @brief Writes the role labels: [N, C] float32, 1 for each class a sample has, else 0.
     *
     * The classes of a sample are the classes of its objects and its class index, when it
     * has one. A sample with neither has a row of zeros.
     */
    class MultiLabelCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator of a set of C = @p class_count classes.
             * @return The collator, never null; InvalidDescription for a count below 1.
             */
            [[nodiscard]] static Result<std::unique_ptr<MultiLabelCollator>> make(
                std::int32_t class_count) noexcept;

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /**
             * @brief Writes the inputs and one row of C flags per sample.
             * @return What Collator::collate states; InvalidDescription for a class outside
             *         [0, class count).
             */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the description of labels over @p class_count classes. */
            explicit MultiLabelCollator(std::int32_t class_count);

            /// @brief Counts the flags of one row, C.
            std::int32_t class_count_;

            /// @brief Holds the description of the labels role.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Vision::Datasets
