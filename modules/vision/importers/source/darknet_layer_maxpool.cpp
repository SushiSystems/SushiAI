/**************************************************************************/
/* darknet_layer_maxpool.cpp                                              */
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
 * @file darknet_layer_maxpool.cpp
 * @brief Implements the translator of a [maxpool] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key a [maxpool] section may hold. */
                constexpr std::string_view K_MAXPOOL_KEYS[] = {"size", "stride", "padding"};

                /** @brief Holds what a [maxpool] section states, defaults filled in. */
                struct Pool
                {
                    std::int64_t size = 1;    /**< Side of the square window. */
                    std::int64_t stride = 1;  /**< Step of the window. */
                    std::int64_t padding = 0; /**< Pad summed over both sides of an axis. */
                };

                /** @brief Holds how one axis of a pool is padded and what it yields. */
                struct PoolAxis
                {
                    std::int64_t extent = 0;   /**< Windows along the axis. */
                    std::int64_t trailing = 0; /**< Pad after the input beyond the leading. */
                };

                /**
                 * @brief Returns the extent of one axis and the trailing pad that yields it.
                 * @param input The extent the pool reads along the axis.
                 * @return The extent `(input + padding - size) / stride + 1` and the pad a row
                 *         with `padding / 2` on both sides still lacks to reach it.
                 */
                [[nodiscard]] PoolAxis fit_axis(const Pool& pool, std::int64_t input) noexcept
                {
                    const std::int64_t leading = pool.padding / 2;
                    PoolAxis axis;
                    axis.extent = (input + pool.padding - pool.size) / pool.stride + 1;

                    const std::int64_t covered = (axis.extent - 1) * pool.stride + pool.size;
                    axis.trailing = std::max<std::int64_t>(0, covered - input - 2 * leading);
                    return axis;
                }

                /** @brief Returns the extent a row padded as @p axis says reports for @p input. */
                [[nodiscard]] std::int64_t row_extent(const Pool& pool, const PoolAxis& axis,
                                                      std::int64_t input) noexcept
                {
                    const std::int64_t padded = input + 2 * (pool.padding / 2) + axis.trailing;
                    return (padded - pool.size) / pool.stride + 1;
                }

            } // namespace

            Status translate_maxpool(const DarknetSection& section, TranslationProgress& progress)
            {
                SA_TRY(reject_unknown_options(section, K_MAXPOOL_KEYS));

                Pool pool;
                SA_TRY_ASSIGN(pool.stride, read_integer(section, "stride", 1));
                SA_TRY_ASSIGN(pool.size, read_integer(section, "size", pool.stride));
                SA_FAIL_IF(pool.size < 1 || pool.stride < 1, StatusCode::InvalidDescription,
                           "line {}: a [maxpool] section needs a positive size and stride; got "
                           "{} and {}",
                           section.line, pool.size, pool.stride);
                SA_TRY_ASSIGN(pool.padding, read_integer(section, "padding", pool.size - 1));
                SA_FAIL_IF(pool.padding < 0, StatusCode::InvalidDescription,
                           "line {}: a [maxpool] section needs a padding of 0 or more; got {}",
                           section.line, pool.padding);

                const LayerValue source = previous_value(progress);
                SA_FAIL_IF(source.height + pool.padding < pool.size ||
                               source.width + pool.padding < pool.size,
                           StatusCode::InvalidDescription,
                           "line {}: a window of {} with padding {} does not fit the {} by {} "
                           "value it reads",
                           section.line, pool.size, pool.padding, source.height, source.width);

                const PoolAxis rows = fit_axis(pool, source.height);
                const PoolAxis columns = fit_axis(pool, source.width);
                SA_FAIL_IF(row_extent(pool, rows, source.height) != rows.extent ||
                               row_extent(pool, columns, source.width) != columns.extent,
                           StatusCode::Unsupported,
                           "line {}: no maxpool2d row gives the {} by {} extent this pool of "
                           "size {}, stride {} and padding {} has over {} by {}",
                           section.line, rows.extent, columns.extent, pool.size, pool.stride,
                           pool.padding, source.height, source.width);

                const std::int64_t sources[] = {source.row};
                const std::int64_t row = append_row(
                    progress, "maxpool2d",
                    make_attributes({"kernel", "stride", "padding", "trailing_padding"},
                                    {Config::Value::integer(pool.size),
                                     Config::Value::integer(pool.stride),
                                     Config::Value::integer(pool.padding / 2),
                                     Config::Value::array(
                                         {Config::Value::integer(rows.trailing),
                                          Config::Value::integer(columns.trailing)})}),
                    sources);

                progress.layers.push_back(
                    LayerValue{row, rows.extent, columns.extent, source.channels});
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
