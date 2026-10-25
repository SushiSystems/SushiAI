/**************************************************************************/
/* darknet_model.cpp                                                      */
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
 * @file darknet_model.cpp
 * @brief Implements the translation of a parsed .cfg, one section at a time through the table.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/darknet_model.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>

#include "darknet_layer_table.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Lists the keys of the first section: those read, then those accepted. */
            constexpr std::string_view K_NETWORK_KEYS[] = {
                "width",      "height",   "channels",      "letter_box", "batch",
                "subdivisions", "momentum", "decay",       "learning_rate", "burn_in",
                "max_batches", "policy",  "steps",         "scales",     "angle",
                "saturation", "exposure", "hue",           "mosaic",
            };

            /**
             * @brief Reads the input's size from the first section into @p progress.
             * @return Unsupported naming the line of an unknown key; InvalidDescription for a
             *         `letter_box` other than 0 or 1, or a width, height or channel count
             *         below 1.
             */
            [[nodiscard]] Status read_network(const DarknetSection& net,
                                              Detail::TranslationProgress& progress)
            {
                SA_TRY(Detail::reject_unknown_options(net, K_NETWORK_KEYS));

                SA_TRY_ASSIGN(const std::int64_t letter_box,
                              Detail::read_integer(net, "letter_box", 0));
                SA_FAIL_IF(letter_box != 0 && letter_box != 1, StatusCode::InvalidDescription,
                           "line {}: letter_box={}; Darknet reads 0 or 1",
                           Detail::find_option(net, "letter_box")->line, letter_box);

                Detail::LayerValue& input = progress.input;
                SA_TRY_ASSIGN(input.width, Detail::read_integer(net, "width", 0));
                SA_TRY_ASSIGN(input.height, Detail::read_integer(net, "height", 0));
                SA_TRY_ASSIGN(input.channels, Detail::read_integer(net, "channels", 0));
                SA_FAIL_IF(input.width < 1 || input.height < 1 || input.channels < 1,
                           StatusCode::InvalidDescription,
                           "line {}: the [{}] section needs a width, a height and channels of 1 "
                           "or more; got {}, {} and {}",
                           net.line, net.type, input.width, input.height, input.channels);

                progress.model.network.height = input.height;
                progress.model.network.width = input.width;
                progress.model.spec.input_shape = Shape{input.height, input.width, input.channels};
                progress.model.spec.input_fit =
                    std::string(letter_box == 1 ? Config::K_FIT_LETTERBOX : Config::K_FIT_RESIZE);
                return Status();
            }

        } // namespace

        Result<DarknetModel> translate_darknet_model(const DarknetConfig& config,
                                                     const DarknetTranslation& translation)
        {
            Detail::TranslationProgress progress;
            progress.normalisation = translation.normalisation;
            SA_TRY(read_network(config.net, progress));

            progress.layers.reserve(config.layers.size());
            for (std::size_t layer = 0; layer < config.layers.size(); ++layer)
            {
                const DarknetSection& section = config.layers[layer];
                const Detail::LayerTableRow* row = Detail::find_layer_translator(section.type);
                SA_FAIL_IF(row == nullptr, StatusCode::Unsupported,
                           "line {}: layer {} is a [{}] section, which this importer does not "
                           "translate",
                           section.line, layer, section.type);

                SA_TRY(row->translate(section, progress));
                SA_FATAL_IF(progress.layers.size() != layer + 1,
                            "the translator of [{}] left {} values after layer {}", section.type,
                            progress.layers.size(), layer);
            }

            DarknetModel& model = progress.model;
            SA_FAIL_IF(model.heads.empty(), StatusCode::InvalidDescription,
                       "the configuration has no [yolo] section, so its model publishes nothing");
            SA_TRY(Detect::Anchor::validate_anchor_layout(model.layout));

            model.row_of_layer.reserve(progress.layers.size());
            for (const Detail::LayerValue& value : progress.layers)
                model.row_of_layer.push_back(value.row);

            return std::move(model);
        }

    } // namespace Vision::Importers
} // namespace SushiAI
