/**************************************************************************/
/* darknet_layer_table.hpp                                                */
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
 * @file darknet_layer_table.hpp
 * @brief Declares the table of section translators and what every translator works with.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the units of the translation that no caller names. */
        namespace Detail
        {
            /** @brief Holds the row that carries a layer's output and the extents of it. */
            struct LayerValue
            {
                std::int64_t row = Config::K_MODEL_INPUT; /**< Row index, or the input. */
                std::int64_t height = 0;                  /**< Rows of one sample. */
                std::int64_t width = 0;                   /**< Columns of one sample. */
                std::int64_t channels = 0;                /**< Channels of one sample. */
            };

            /** @brief Holds a translation under way: the model so far and each layer's value. */
            struct TranslationProgress
            {
                /** @brief Holds the form of every normalised convolution. */
                DarknetNormalisation normalisation = DarknetNormalisation::Separate;

                /** @brief Holds the network input, which layer 0 reads. */
                LayerValue input;

                /** @brief Holds the value of every layer translated so far, by layer index. */
                std::vector<LayerValue> layers;

                /** @brief Holds the model the translators append to. */
                DarknetModel model;
            };

            /**
             * @brief Translates one section and appends its value to progress.layers.
             * @return Success after exactly one value was appended, or a failure naming the
             *         line at fault.
             */
            using LayerTranslator = Status (*)(const DarknetSection& section,
                                               TranslationProgress& progress);

            /** @brief Holds one row of the table: a section type and its translator. */
            struct LayerTableRow
            {
                std::string_view type;     /**< The name between the brackets. */
                LayerTranslator translate; /**< The translator of that section type. */
            };

            /** @brief Returns the translator row of @p type, or null when none holds it. */
            [[nodiscard]] const LayerTableRow* find_layer_translator(
                std::string_view type) noexcept;

            /** @brief Translates a [convolutional] section into two or three rows. */
            [[nodiscard]] Status translate_convolutional(const DarknetSection& section,
                                                         TranslationProgress& progress);

            /** @brief Translates a [maxpool] section into a maxpool2d row. */
            [[nodiscard]] Status translate_maxpool(const DarknetSection& section,
                                                   TranslationProgress& progress);

            /** @brief Translates an [upsample] section into an upsample row. */
            [[nodiscard]] Status translate_upsample(const DarknetSection& section,
                                                    TranslationProgress& progress);

            /** @brief Translates a [route] section into a source or a concat row. */
            [[nodiscard]] Status translate_route(const DarknetSection& section,
                                                 TranslationProgress& progress);

            /** @brief Translates a [shortcut] section into an add row of two sources. */
            [[nodiscard]] Status translate_shortcut(const DarknetSection& section,
                                                    TranslationProgress& progress);

            /** @brief Translates a [yolo] section into a head, its anchors and an output. */
            [[nodiscard]] Status translate_yolo(const DarknetSection& section,
                                                TranslationProgress& progress);

            /** @brief Returns Unsupported naming the first key of @p section not in @p known. */
            [[nodiscard]] Status reject_unknown_options(
                const DarknetSection& section, SushiRuntime::span<const std::string_view> known);

            /** @brief Returns the option of @p section spelled @p key, or null when absent. */
            [[nodiscard]] const DarknetOption* find_option(const DarknetSection& section,
                                                           std::string_view key) noexcept;

            /**
             * @brief Reads the key @p key as a whole number, or returns @p fallback when absent.
             * @return InvalidDescription naming the line when the value is not a whole number
             *         of 32 bits.
             */
            [[nodiscard]] Result<std::int64_t> read_integer(const DarknetSection& section,
                                                            std::string_view key,
                                                            std::int64_t fallback);

            /**
             * @brief Reads the key @p key as a finite number, or returns @p fallback when absent.
             * @return InvalidDescription naming the line when the value is not a finite number.
             */
            [[nodiscard]] Result<double> read_real(const DarknetSection& section,
                                                   std::string_view key, double fallback);

            /**
             * @brief Reads the key @p key as a comma-separated list of whole numbers.
             * @return InvalidDescription naming the line of an entry that is not a whole number
             *         of 32 bits, or the section's line when the key is absent or empty.
             */
            [[nodiscard]] Result<std::vector<std::int64_t>> read_integer_list(
                const DarknetSection& section, std::string_view key);

            /**
             * @brief Reads the key @p key as a comma-separated list of finite numbers.
             * @return InvalidDescription naming the line of an entry that is not a finite
             *         number, or the section's line when the key is absent or empty.
             */
            [[nodiscard]] Result<std::vector<double>> read_real_list(
                const DarknetSection& section, std::string_view key);

            /** @brief Returns the index of the layer being translated. */
            [[nodiscard]] std::int64_t current_layer(const TranslationProgress& progress) noexcept;

            /** @brief Returns the value of the layer before the current one, or the input. */
            [[nodiscard]] const LayerValue& previous_value(
                const TranslationProgress& progress) noexcept;

            /**
             * @brief Returns the value of the layer a route or shortcut entry names.
             * @param reference Below zero it counts back from the current layer; else absolute.
             * @return InvalidDescription naming @p line when no earlier layer has that index.
             */
            [[nodiscard]] Result<LayerValue> resolve_layer(const TranslationProgress& progress,
                                                           std::int64_t reference,
                                                           std::int64_t line);

            /** @brief Returns an object value holding @p keys and @p values, matched by index. */
            [[nodiscard]] Config::Value make_attributes(std::vector<std::string> keys,
                                                        std::vector<Config::Value> values);

            /**
             * @brief Appends a row of @p type that reads @p sources and returns its index.
             * @param sources Row indices or Config::K_MODEL_INPUT, in the order the row reads.
             */
            std::int64_t append_row(TranslationProgress& progress, std::string_view type,
                                    Config::Value attributes,
                                    SushiRuntime::span<const std::int64_t> sources);

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
