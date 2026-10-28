/**************************************************************************/
/* binary_collator.hpp                                                    */
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
 * @file binary_collator.hpp
 * @brief Declares the collator of the binary task.
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
    /** @brief Names the binary task in a training file. */
    inline constexpr std::string_view K_TASK_BINARY = "binary";

    /** @brief Counts the classes of a set the binary task reads. */
    inline constexpr std::int32_t K_BINARY_CLASSES = 2;

    /** @brief Writes the role labels: one float32 per sample, 1 or 0, of shape [N]. */
    class BinaryCollator final : public Collator
    {
        public:
            /**
             * @brief Builds the collator that writes 1 for the class @p positive_class.
             * @return The collator, never null; InvalidDescription for a class that is
             *         neither 0 nor 1.
             */
            [[nodiscard]] static Result<std::unique_ptr<BinaryCollator>> make(
                std::int32_t positive_class) noexcept;

            [[nodiscard]] std::string_view task() const noexcept override;
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override;

            /**
             * @brief Writes the inputs, and 1 for a sample of the positive class, else 0.
             * @return What Collator::collate states; InvalidDescription for a sample with no
             *         class index or with one outside [0, 2).
             */
            [[nodiscard]] Status collate(const SampleList& list,
                                         SushiRuntime::span<const FittedSample> samples,
                                         Data::Batch& batch) const override;

        private:
            /** @brief Holds the description of labels and the class that is written as 1. */
            explicit BinaryCollator(std::int32_t positive_class);

            /// @brief Holds the class index written as 1.
            std::int32_t positive_class_;

            /// @brief Holds the description of the labels role.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Vision::Datasets
