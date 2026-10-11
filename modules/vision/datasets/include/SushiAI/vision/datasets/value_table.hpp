/**************************************************************************/
/* value_table.hpp                                                        */
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
 * @file value_table.hpp
 * @brief Declares the table of real numbers a tabular sample list keeps, one row per sample.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <SushiRuntime/core/span.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Holds the features and the targets of every sample of a tabular list.
     *
     * Row i belongs to sample i and holds feature_count features, then target_count
     * targets. A list of images holds a table of no row.
     */
    struct ValueTable
    {
        /** @brief Names the columns of a row: the features, then the targets. */
        std::vector<std::string> columns;

        /** @brief Counts the features of one row. */
        std::int32_t feature_count = 0;

        /** @brief Counts the targets of one row. */
        std::int32_t target_count = 0;

        /** @brief Holds the rows one after another. */
        std::vector<float> values;

        /** @brief Returns the numbers one row holds. */
        [[nodiscard]] std::size_t row_width() const noexcept
        {
            return static_cast<std::size_t>(feature_count) +
                   static_cast<std::size_t>(target_count);
        }

        /** @brief Returns the whole rows the table holds; 0 for a table of no column. */
        [[nodiscard]] std::size_t row_count() const noexcept
        {
            return row_width() == 0 ? 0 : values.size() / row_width();
        }

        /**
         * @brief Returns the features of one row.
         * @param row Below row_count(); not checked.
         */
        [[nodiscard]] SushiRuntime::span<const float> features(std::size_t row) const noexcept
        {
            return SushiRuntime::span<const float>(values.data() + row * row_width(),
                                                   static_cast<std::size_t>(feature_count));
        }

        /**
         * @brief Returns the targets of one row.
         * @param row Below row_count(); not checked.
         */
        [[nodiscard]] SushiRuntime::span<const float> targets(std::size_t row) const noexcept
        {
            return SushiRuntime::span<const float>(
                values.data() + row * row_width() + static_cast<std::size_t>(feature_count),
                static_cast<std::size_t>(target_count));
        }
    };
} // namespace SushiAI::Vision::Datasets
