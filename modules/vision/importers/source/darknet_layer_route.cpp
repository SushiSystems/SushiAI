/**************************************************************************/
/* darknet_layer_route.cpp                                                */
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
 * @file darknet_layer_route.cpp
 * @brief Implements the translator of a [route] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
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
                /** @brief Lists every key a [route] section may hold. */
                constexpr std::string_view K_ROUTE_KEYS[] = {"layers", "groups", "group_id"};

                /** @brief Holds the key that lists the layers a route joins. */
                constexpr std::string_view K_LAYERS_KEY = "layers";

                /**
                 * @brief Returns the group @p group_id of @p source, appending a slice row when
                 *        @p groups is above 1.
                 * @return @p source itself when @p groups is 1; otherwise a value of
                 *         C / groups channels, or InvalidDescription naming @p line when
                 *         @p groups does not divide the channels of @p source.
                 */
                [[nodiscard]] Result<LayerValue> select_group(TranslationProgress& progress,
                                                const LayerValue& source, std::int64_t groups,
                                                std::int64_t group_id, std::int64_t line)
                {
                    if (groups == 1)
                        return source;

                    SA_FAIL_IF(source.channels % groups != 0, StatusCode::InvalidDescription,
                               "line {}: a [route] section with groups {} cannot split a value "
                               "of {} channels into equal groups",
                               line, groups, source.channels);

                    const std::int64_t count = source.channels / groups;
                    const std::int64_t offset = group_id * count;
                    const std::int64_t sources[] = {source.row};
                    const std::int64_t row = append_row(
                        progress, "slice_channels",
                        make_attributes({"offset", "count"}, {Config::Value::integer(offset),
                                                              Config::Value::integer(count)}),
                        sources);

                    return LayerValue{row, source.height, source.width, count};
                }

            } // namespace

            Status translate_route(const DarknetSection& section, TranslationProgress& progress)
            {
                SA_TRY(reject_unknown_options(section, K_ROUTE_KEYS));

                SA_TRY_ASSIGN(const std::int64_t groups, read_integer(section, "groups", 1));
                SA_TRY_ASSIGN(const std::int64_t group_id, read_integer(section, "group_id", 0));
                SA_FAIL_IF(groups < 1 || group_id < 0 || group_id >= groups,
                           StatusCode::InvalidDescription,
                           "line {}: a [route] section needs 1 <= groups and 0 <= group_id < "
                           "groups; got groups {} and group_id {}",
                           section.line, groups, group_id);

                SA_TRY_ASSIGN(const std::vector<std::int64_t> references,
                              read_integer_list(section, K_LAYERS_KEY));
                const std::int64_t line = find_option(section, K_LAYERS_KEY)->line;

                std::vector<std::int64_t> rows;
                rows.reserve(references.size());
                LayerValue joined;
                for (std::int64_t reference : references)
                {
                    SA_TRY_ASSIGN(const LayerValue source,
                                  resolve_layer(progress, reference, line));
                    if (rows.empty())
                    {
                        joined = source;
                        joined.channels = 0;
                    }
                    SA_FAIL_IF(source.height != joined.height || source.width != joined.width,
                               StatusCode::InvalidDescription,
                               "line {}: a [route] section joins a {} by {} value with a {} by "
                               "{} one; the layers it lists must share both extents",
                               line, joined.height, joined.width, source.height, source.width);

                    SA_TRY_ASSIGN(const LayerValue part,
                                  select_group(progress, source, groups, group_id, line));
                    joined.channels += part.channels;
                    rows.push_back(part.row);
                }

                // A route of one part adds no row: the layers after it read that part's row.
                joined.row = rows.front();
                if (rows.size() > 1)
                {
                    joined.row = append_row(
                        progress, "concat", make_attributes({}, {}),
                        SushiRuntime::span<const std::int64_t>(rows.data(), rows.size()));
                }

                progress.layers.push_back(joined);
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
