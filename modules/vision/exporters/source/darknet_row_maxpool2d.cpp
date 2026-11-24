/**************************************************************************/
/* darknet_row_maxpool2d.cpp                                              */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file darknet_row_maxpool2d.cpp
 * @brief Defines the writer of the "maxpool2d" row, one [maxpool] section.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include "darknet_export_progress.hpp"
#include "darknet_row_table.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key of the row the writer reads. */
                constexpr std::string_view K_MAXPOOL2D_KEYS[] = {"kernel", "stride", "padding",
                                                                 "trailing_padding"};

                /** @brief Holds a [maxpool] section: a square window and the padding total. */
                struct Pool
                {
                    std::int64_t size = 1;    /**< Side of the window. */
                    std::int64_t stride = 1;  /**< Step of the window. */
                    std::int64_t padding = 0; /**< Pad summed over both sides of an axis. */
                };

                /**
                 * @brief Returns the extent the importer gives one axis of @p input under
                 *        @p pool, or -1 when it gives back no row of @p trailing.
                 */
                std::int64_t fit_axis(const Pool& pool, std::int64_t input,
                                      std::int64_t trailing) noexcept
                {
                    if (input + pool.padding < pool.size)
                        return -1;
                    const std::int64_t leading = pool.padding / 2;
                    const std::int64_t extent =
                        (input + pool.padding - pool.size) / pool.stride + 1;
                    const std::int64_t covered = (extent - 1) * pool.stride + pool.size;
                    const std::int64_t found =
                        std::max<std::int64_t>(0, covered - input - 2 * leading);
                    const std::int64_t padded = input + 2 * leading + found;
                    if (found != trailing || (padded - pool.size) / pool.stride + 1 != extent)
                        return -1;
                    return extent;
                }
            } // namespace

            Status write_darknet_maxpool2d(const DarknetRowCall& call,
                                           DarknetExportProgress& progress)
            {
                const Config::LayerSpec& spec = *call.spec;
                const RowKeys keys = row_keys(spec);
                SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                     K_MAXPOOL2D_KEYS,
                                                     std::size(K_MAXPOOL2D_KEYS))));
                SA_TRY(check_source_count(call, 1));
                Pool pool;
                SA_TRY_ASSIGN(pool.size, read_square_key(keys, "kernel", 1));
                pool.stride = pool.size;
                if (spec.attributes.has("stride"))
                {
                    SA_TRY_ASSIGN(pool.stride, read_square_key(keys, "stride", 1));
                }
                SA_TRY_ASSIGN(const std::int64_t leading, read_square_key(keys, "padding", 0));
                SA_TRY_ASSIGN(const std::vector<std::int64_t> trailing,
                              read_pair_key(keys, "trailing_padding", 0));
                SA_FAIL_IF(pool.size < 1 || pool.stride < 1 || leading < 0,
                           StatusCode::InvalidDescription,
                           "row {} 'maxpool2d' states a kernel of {}, a stride of {} and a "
                           "padding of {}",
                           call.index, pool.size, pool.stride, leading);
                const std::vector<std::int64_t>& input = call.values[0]->shape;
                // The importer halves the total, so a total of 2p or 2p + 1 gives back p.
                std::int64_t height = -1;
                std::int64_t width = -1;
                for (pool.padding = 2 * leading; pool.padding <= 2 * leading + 1; ++pool.padding)
                {
                    height = fit_axis(pool, input[0], trailing[0]);
                    width = fit_axis(pool, input[1], trailing[1]);
                    if (height >= 0 && width >= 0)
                        break;
                }
                SA_FAIL_IF(height < 0 || width < 0, StatusCode::Unsupported,
                           "its trailing padding of {} by {} is not the one the Darknet "
                           "importer gives a [maxpool] of padding {} or {}",
                           trailing[0], trailing[1], 2 * leading, 2 * leading + 1);
                SA_TRY(follow_row(progress, call.sources[0]));
                add_section(progress, "maxpool",
                            {DarknetOptionText{"size", spell_integer(pool.size)},
                             DarknetOptionText{"stride", spell_integer(pool.stride)},
                             DarknetOptionText{"padding", spell_integer(pool.padding)}});
                define_row_value(progress, call.index, {height, width, input[2]});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
