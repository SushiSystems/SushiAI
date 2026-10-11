/**************************************************************************/
/* binary_confusion.cpp                                                   */
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
 * @file binary_confusion.cpp
 * @brief Implements the count of right and wrong yes-or-no predictions and the ratios over it.
 * @author Mustafa Garip
 */

#include <SushiAI/train/binary_confusion.hpp>

#include <SushiAI/train/host_values.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Holds the truth value above which an element reads as 1. */
            constexpr double K_TRUTH_CUT = 0.5;

            /** @brief Returns @p part / @p whole, or 0 when @p whole is 0. */
            [[nodiscard]] double ratio(std::int64_t part, std::int64_t whole) noexcept
            {
                return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
            }
        } // namespace

        BinaryConfusion count_confusion(const SushiBLAS::Tensor& logits,
                                        const SushiBLAS::Tensor& truth,
                                        const ColumnOfLogits& where, double cut) noexcept
        {
            BinaryConfusion counts;
            for (std::int64_t row = 0; row < where.rows; ++row)
            {
                const std::int64_t index = row * where.columns + where.column;
                const bool predicted = host_real(logits, index) >= cut;
                const bool actual = host_real(truth, index) > K_TRUTH_CUT;

                if (predicted && actual)
                    ++counts.true_positives;
                else if (predicted)
                    ++counts.false_positives;
                else if (actual)
                    ++counts.false_negatives;
                else
                    ++counts.true_negatives;
            }

            return counts;
        }

        double accuracy_of(const BinaryConfusion& counts) noexcept
        {
            const std::int64_t right = counts.true_positives + counts.true_negatives;

            return ratio(right, right + counts.false_positives + counts.false_negatives);
        }

        double precision_of(const BinaryConfusion& counts) noexcept
        {
            return ratio(counts.true_positives, counts.true_positives + counts.false_positives);
        }

        double recall_of(const BinaryConfusion& counts) noexcept
        {
            return ratio(counts.true_positives, counts.true_positives + counts.false_negatives);
        }

        double f1_of(const BinaryConfusion& counts) noexcept
        {
            return ratio(2 * counts.true_positives,
                         2 * counts.true_positives + counts.false_positives
                             + counts.false_negatives);
        }

    } // namespace Train
} // namespace SushiAI
