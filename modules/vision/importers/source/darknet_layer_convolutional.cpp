/**************************************************************************/
/* darknet_layer_convolutional.cpp                                        */
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
 * @file darknet_layer_convolutional.cpp
 * @brief Implements the translator of a [convolutional] section.
 * @author Mustafa Garip
 */

#include "darknet_layer_table.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists every key a [convolutional] section may hold. */
                constexpr std::string_view K_CONVOLUTIONAL_KEYS[] = {
                    "filters", "size", "stride", "pad", "padding", "batch_normalize", "activation"};

                /** @brief Holds the activation of a section that names none. */
                constexpr std::string_view K_DEFAULT_ACTIVATION = "logistic";

                /** @brief Holds the epsilon inside the root of the file's normalisation. */
                constexpr double K_NORMALISATION_EPSILON = 1e-5;

                /** @brief Holds one activation name and the row it becomes. */
                struct ActivationRow
                {
                    std::string_view name; /**< The value of the section's activation key. */
                    std::string_view type; /**< The row type; empty when no row follows. */
                    bool has_slope;        /**< Whether the row carries a slope key. */
                    double slope;          /**< The slope below zero, read with has_slope. */
                };

                /** @brief Lists every activation this importer translates. */
                constexpr ActivationRow K_ACTIVATIONS[] = {
                    {"leaky", "leaky_relu", true, 0.1}, {"mish", "mish", false, 0.0},
                    {"swish", "silu", false, 0.0},      {"logistic", "sigmoid", false, 0.0},
                    {"linear", "", false, 0.0},
                };

                /** @brief Returns the activation row of @p name, or null when none holds it. */
                [[nodiscard]] const ActivationRow* find_activation(std::string_view name) noexcept
                {
                    for (const ActivationRow& row : K_ACTIVATIONS)
                    {
                        if (row.name == name)
                            return &row;
                    }
                    return nullptr;
                }

                /** @brief Holds what a [convolutional] section states, defaults filled in. */
                struct Convolution
                {
                    std::int64_t filters = 1;                /**< Output channels. */
                    std::int64_t size = 1;                   /**< Side of the square kernel. */
                    std::int64_t stride = 1;                 /**< Step of the window. */
                    std::int64_t padding = 0;                /**< Pad on each of the four sides. */
                    bool normalised = false;                 /**< Whether batch norm follows. */
                    const ActivationRow* activation = nullptr; /**< The activation's row. */
                };

                /**
                 * @brief Reads the keys of a [convolutional] section over their defaults.
                 * @return Unsupported for an unknown key or activation; InvalidDescription for
                 *         a value that is malformed or out of range. Each names its line.
                 */
                [[nodiscard]] Result<Convolution> read_convolution(const DarknetSection& section)
                {
                    SA_TRY(reject_unknown_options(section, K_CONVOLUTIONAL_KEYS));

                    Convolution read;
                    SA_TRY_ASSIGN(read.filters, read_integer(section, "filters", 1));
                    SA_TRY_ASSIGN(read.size, read_integer(section, "size", 1));
                    SA_TRY_ASSIGN(read.stride, read_integer(section, "stride", 1));
                    SA_TRY_ASSIGN(const std::int64_t pad, read_integer(section, "pad", 0));
                    SA_TRY_ASSIGN(const std::int64_t padding, read_integer(section, "padding", 0));
                    SA_TRY_ASSIGN(const std::int64_t normalise,
                                  read_integer(section, "batch_normalize", 0));
                    SA_FAIL_IF(read.filters < 1 || read.size < 1 || read.stride < 1 || padding < 0,
                               StatusCode::InvalidDescription,
                               "line {}: a [convolutional] section needs positive filters, size "
                               "and stride and a padding of 0 or more; got {}, {}, {} and {}",
                               section.line, read.filters, read.size, read.stride, padding);

                    read.padding = pad != 0 ? read.size / 2 : padding;
                    read.normalised = normalise != 0;

                    const DarknetOption* named = find_option(section, "activation");
                    const std::string_view name =
                        named == nullptr ? K_DEFAULT_ACTIVATION : std::string_view(named->value);
                    read.activation = find_activation(name);
                    SA_FAIL_IF(read.activation == nullptr, StatusCode::Unsupported,
                               "line {}: the activation '{}' is not one this importer translates",
                               named == nullptr ? section.line : named->line, name);

                    return read;
                }

                /** @brief Returns the state entry @p field of row @p row, "<row>.<field>". */
                [[nodiscard]] std::string entry_name(std::int64_t row, std::string_view field)
                {
                    return std::to_string(row).append(".").append(field);
                }

                /** @brief Appends the conv2d row of @p read over @p source; returns its index. */
                std::int64_t append_convolution(TranslationProgress& progress,
                                                const Convolution& read, std::int64_t source,
                                                bool use_bias)
                {
                    const std::int64_t sources[] = {source};
                    return append_row(
                        progress, "conv2d",
                        make_attributes({"filters", "kernel", "stride", "padding", "use_bias"},
                                        {Config::Value::integer(read.filters),
                                         Config::Value::integer(read.size),
                                         Config::Value::integer(read.stride),
                                         Config::Value::integer(read.padding),
                                         Config::Value::boolean(use_bias)}),
                        sources);
                }

                /** @brief Appends the batchnorm2d row over @p source; returns its index. */
                std::int64_t append_normalisation(TranslationProgress& progress,
                                                  std::int64_t source)
                {
                    const std::int64_t sources[] = {source};
                    return append_row(
                        progress, "batchnorm2d",
                        make_attributes({"bn_epsilon"},
                                        {Config::Value::real(K_NORMALISATION_EPSILON)}),
                        sources);
                }

                /** @brief Appends the row of @p activation over @p source; returns its index. */
                std::int64_t append_activation(TranslationProgress& progress,
                                               const ActivationRow& activation,
                                               std::int64_t source)
                {
                    const std::int64_t sources[] = {source};
                    Config::Value attributes =
                        activation.has_slope
                            ? make_attributes({"slope"}, {Config::Value::real(activation.slope)})
                            : make_attributes({}, {});
                    return append_row(progress, activation.type, std::move(attributes), sources);
                }

            } // namespace

            Status translate_convolutional(const DarknetSection& section,
                                           TranslationProgress& progress)
            {
                SA_TRY_ASSIGN(const Convolution read, read_convolution(section));

                const LayerValue source = previous_value(progress);
                const std::int64_t span_h = source.height + 2 * read.padding - read.size;
                const std::int64_t span_w = source.width + 2 * read.padding - read.size;
                SA_FAIL_IF(span_h < 0 || span_w < 0, StatusCode::InvalidDescription,
                           "line {}: a kernel of {} with padding {} does not fit the {} by {} "
                           "value it reads",
                           section.line, read.size, read.padding, source.height, source.width);

                const bool separate = read.normalised &&
                                      progress.normalisation == DarknetNormalisation::Separate;
                const std::int64_t convolution =
                    append_convolution(progress, read, source.row, !separate);

                DarknetWeightTarget target;
                target.weight = entry_name(convolution, "weight");
                std::int64_t last = convolution;
                if (separate)
                {
                    last = append_normalisation(progress, convolution);
                    target.bias = entry_name(last, "bias");
                    target.scale = entry_name(last, "weight");
                    target.mean = entry_name(last, "running_mean");
                    target.variance = entry_name(last, "running_var");
                }
                else
                {
                    target.bias = entry_name(convolution, "bias");
                }

                if (!read.activation->type.empty())
                    last = append_activation(progress, *read.activation, last);

                DarknetConvolutionShape shape;
                shape.layer = current_layer(progress);
                shape.filters = read.filters;
                shape.channels = source.channels;
                shape.size = read.size;
                shape.normalised = read.normalised;
                progress.model.convolutions.push_back(shape);
                progress.model.targets.push_back(std::move(target));

                progress.layers.push_back(LayerValue{last, span_h / read.stride + 1,
                                                     span_w / read.stride + 1, read.filters});
                return Status();
            }

        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
