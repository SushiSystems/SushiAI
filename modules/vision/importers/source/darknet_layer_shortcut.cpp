/**************************************************************************/
/* darknet_layer_shortcut.cpp                                             */
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
 * @file darknet_layer_shortcut.cpp
 * @brief Implements the translator of a [shortcut] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
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
                /** @brief Lists every key a [shortcut] section may hold. */
                constexpr std::string_view K_SHORTCUT_KEYS[] = {"from", "activation"};

                /** @brief Holds the key that names the layer a shortcut adds. */
                constexpr std::string_view K_FROM_KEY = "from";

                /** @brief Holds the one activation a shortcut is translated with. */
                constexpr std::string_view K_LINEAR_ACTIVATION = "linear";

                /** @brief Holds the number of layers a shortcut's 'from' names. */
                constexpr std::size_t K_FROM_COUNT = 1;

            } // namespace

            Status translate_shortcut(const DarknetSection& section, TranslationProgress& progress)
            {
                SA_TRY(reject_unknown_options(section, K_SHORTCUT_KEYS));

                SA_TRY_ASSIGN(const std::vector<std::int64_t> references,
                              read_integer_list(section, K_FROM_KEY));
                const std::int64_t line = find_option(section, K_FROM_KEY)->line;
                SA_FAIL_IF(references.size() != K_FROM_COUNT, StatusCode::InvalidDescription,
                           "line {}: a [shortcut] section names one layer in 'from'; it names {}",
                           line, references.size());

                const DarknetOption* activation = find_option(section, "activation");
                SA_FAIL_IF(activation != nullptr && activation->value != K_LINEAR_ACTIVATION,
                           StatusCode::Unsupported,
                           "line {}: a [shortcut] section has the activation '{}'; only linear "
                           "is translated",
                           activation->line, activation->value);

                // An add row takes equal shapes, so the two layers are compared before it.
                const LayerValue before = previous_value(progress);
                SA_TRY_ASSIGN(const LayerValue named, resolve_layer(progress, references[0], line));
                SA_FAIL_IF(named.height != before.height || named.width != before.width ||
                               named.channels != before.channels,
                           StatusCode::InvalidDescription,
                           "line {}: a [shortcut] section adds a {} by {} by {} value to a {} by "
                           "{} by {} one; the two layers must share their shape",
                           line, named.height, named.width, named.channels, before.height,
                           before.width, before.channels);

                const std::int64_t sources[] = {before.row, named.row};
                LayerValue summed = before;
                summed.row = append_row(progress, "add", make_attributes({}, {}),
                                        SushiRuntime::span<const std::int64_t>(sources, 2));

                progress.layers.push_back(summed);
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
