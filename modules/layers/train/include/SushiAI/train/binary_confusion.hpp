/**************************************************************************/
/* binary_confusion.hpp                                                   */
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
 * @file binary_confusion.hpp
 * @brief Declares the count of right and wrong yes-or-no predictions and the ratios over it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <SushiBLAS/tensor/tensor.hpp>

namespace SushiAI
{
    namespace Train
    {
        /** @brief Counts the predictions of one yes-or-no decision against the truth. */
        struct BinaryConfusion
        {
            std::int64_t true_positives = 0;  /**< Predicted 1 where the truth is 1. */
            std::int64_t false_positives = 0; /**< Predicted 1 where the truth is 0. */
            std::int64_t false_negatives = 0; /**< Predicted 0 where the truth is 1. */
            std::int64_t true_negatives = 0;  /**< Predicted 0 where the truth is 0. */
        };

        /** @brief States which column of [rows, columns] logits a count reads. */
        struct ColumnOfLogits
        {
            std::int64_t rows = 0;    /**< Rows of the logits and of the truth. */
            std::int64_t columns = 0; /**< Columns of the logits and of the truth. */
            std::int64_t column = 0;  /**< The column counted, below @c columns. */
        };

        /**
         * @brief Counts one column: 1 is predicted where the logit is at least @p cut.
         * @param truth Real values shaped like @p logits; above one half reads as 1.
         * @pre Both tensors hold rows * columns float32 or float64 elements the host reads.
         */
        [[nodiscard]] BinaryConfusion count_confusion(const SushiBLAS::Tensor& logits,
                                                      const SushiBLAS::Tensor& truth,
                                                      const ColumnOfLogits& where,
                                                      double cut) noexcept;

        /** @brief Returns the fraction of predictions that are right, or 0 for no prediction. */
        [[nodiscard]] double accuracy_of(const BinaryConfusion& counts) noexcept;

        /** @brief Returns the fraction of predicted 1s that are right, or 0 for none predicted. */
        [[nodiscard]] double precision_of(const BinaryConfusion& counts) noexcept;

        /** @brief Returns the fraction of true 1s that were predicted, or 0 for no true 1. */
        [[nodiscard]] double recall_of(const BinaryConfusion& counts) noexcept;

        /**
         * @brief Returns 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall.
         * @return 0 when no 1 is predicted and none is true.
         */
        [[nodiscard]] double f1_of(const BinaryConfusion& counts) noexcept;

    } // namespace Train
} // namespace SushiAI
