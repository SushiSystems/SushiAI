/**************************************************************************/
/* darknet_layer_upsample.cpp                                             */
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
 * @file darknet_layer_upsample.cpp
 * @brief Implements the translator of an [upsample] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

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
                /** @brief Lists every key an [upsample] section may hold. */
                constexpr std::string_view K_UPSAMPLE_KEYS[] = {"stride", "scale"};

            } // namespace

            Status translate_upsample(const DarknetSection& section, TranslationProgress& progress)
            {
                SA_TRY(reject_unknown_options(section, K_UPSAMPLE_KEYS));

                SA_TRY_ASSIGN(const std::int64_t stride, read_integer(section, "stride", 2));
                SA_FAIL_IF(stride < 1, StatusCode::InvalidDescription,
                           "line {}: an [upsample] section needs a positive stride; got {}",
                           section.line, stride);

                SA_TRY_ASSIGN(const double scale, read_real(section, "scale", 1.0));
                SA_FAIL_IF(scale != 1.0, StatusCode::Unsupported,
                           "line {}: an [upsample] section with scale {} is not translated; "
                           "only a scale of 1 is",
                           section.line, scale);

                const LayerValue source = previous_value(progress);
                const std::int64_t sources[] = {source.row};
                const std::int64_t row =
                    append_row(progress, "upsample",
                               make_attributes({"scale"}, {Config::Value::integer(stride)}),
                               sources);

                progress.layers.push_back(LayerValue{row, source.height * stride,
                                                     source.width * stride, source.channels});
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
