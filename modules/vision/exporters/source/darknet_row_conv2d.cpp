/**************************************************************************/
/* darknet_row_conv2d.cpp                                                 */
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
 * @file darknet_row_conv2d.cpp
 * @brief Defines the writer of the "conv2d" row, one [convolutional] section with the batch
 *        normalisation and the activation that follow it.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/filter_layout.hpp>
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
                constexpr std::string_view K_CONV2D_KEYS[] = {
                    "filters", "kernel", "stride", "padding", "dilation", "use_bias",
                    "trailing_padding", "groups"};

                /** @brief Lists the one key of a batchnorm2d row the writer reads. */
                constexpr std::string_view K_BATCHNORM_KEYS[] = {"bn_epsilon"};

                /** @brief Lists the one key of a leaky_relu row the writer reads. */
                constexpr std::string_view K_LEAKY_KEYS[] = {"slope"};

                /** @brief Holds the epsilon the importer gives a [convolutional]'s batch norm. */
                constexpr double K_DARKNET_EPSILON = 1e-5;

                /** @brief Holds the one slope of Darknet's leaky activation. */
                constexpr double K_LEAKY_SLOPE = 0.1;

                /** @brief Holds the rank of a kernel record, [out, kh, kw, in]. */
                constexpr std::size_t K_KERNEL_RANK = 4;

                /** @brief Holds one activation row type and the name a .cfg gives it. */
                struct ActivationName
                {
                    std::string_view type; /**< The row type. */
                    std::string_view name; /**< The value of the activation key. */
                };

                /** @brief Lists every activation row a [convolutional] section states. */
                constexpr ActivationName K_ACTIVATIONS[] = {
                    {"leaky_relu", "leaky"},
                    {"mish", "mish"},
                    {"silu", "swish"},
                    {"sigmoid", "logistic"},
                };

                /** @brief Holds what the row states, read and checked. */
                struct Convolution
                {
                    std::int64_t filters = 0; /**< Output channels. */
                    std::int64_t size = 1;    /**< Side of the square kernel. */
                    std::int64_t stride = 1;  /**< Step of the window. */
                    std::int64_t padding = 0; /**< Pad on each of the four sides. */
                    bool use_bias = true;     /**< Whether the row adds a bias. */
                };

                /** @brief Returns the row's keys; Unsupported for what Darknet cannot state. */
                Result<Convolution> read_convolution(const Config::LayerSpec& spec)
                {
                    const RowKeys keys = row_keys(spec);
                    SA_TRY(reject_unknown_keys(keys, SushiRuntime::span<const std::string_view>(
                                                         K_CONV2D_KEYS, std::size(K_CONV2D_KEYS))));
                    Convolution read;
                    SA_TRY_ASSIGN(read.filters, read_int_key(keys, "filters", 0));
                    SA_TRY_ASSIGN(read.size, read_square_key(keys, "kernel", 1));
                    SA_TRY_ASSIGN(read.stride, read_square_key(keys, "stride", 1));
                    SA_TRY_ASSIGN(read.padding, read_square_key(keys, "padding", 0));
                    SA_TRY_ASSIGN(read.use_bias, read_bool_key(keys, "use_bias", true));
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> trailing,
                                  read_pair_key(keys, "trailing_padding", 0));
                    SA_FAIL_IF(trailing[0] != 0 || trailing[1] != 0, StatusCode::Unsupported,
                               "it states a trailing padding, and Darknet pads both sides alike");
                    SA_TRY_ASSIGN(const std::vector<std::int64_t> dilation,
                                  read_pair_key(keys, "dilation", 1));
                    SA_FAIL_IF(dilation[0] != 1 || dilation[1] != 1, StatusCode::Unsupported,
                               "it states a dilation of {} by {}, and the Darknet importer reads "
                               "a dilation of 1 only",
                               dilation[0], dilation[1]);
                    SA_TRY_ASSIGN(const std::int64_t groups, read_int_key(keys, "groups", 1));
                    SA_FAIL_IF(groups != 1, StatusCode::Unsupported,
                               "it states {} groups, and the Darknet importer reads no groups",
                               groups);
                    SA_FAIL_IF(read.filters < 1 || read.size < 1 || read.stride < 1 ||
                                   read.padding < 0,
                               StatusCode::InvalidDescription,
                               "the row states {} filters, a kernel of {}, a stride of {} and a "
                               "padding of {}",
                               read.filters, read.size, read.stride, read.padding);
                    return read;
                }

                /**
                 * @brief Returns the row after @p row when it reads @p row alone and is its one
                 *        reader, and @p row is not published; else null.
                 */
                const Config::LayerSpec* sole_follower(const DarknetExportProgress& progress,
                                                       std::size_t row)
                {
                    const std::vector<Config::LayerSpec>& layers = progress.model->model.layers;
                    const std::size_t next = row + 1;
                    if (next >= layers.size() || progress.published[row] ||
                        progress.readers[row].size() != 1 || progress.readers[row][0] != next)
                        return nullptr;
                    const std::vector<Config::SourceSpec>& from = layers[next].from;
                    if (from.size() != 1 || static_cast<std::size_t>(from[0].node) != row ||
                        !(from[0].port.empty() || from[0].port == "output"))
                        return nullptr;
                    return &layers[next];
                }

                /** @brief Returns the .cfg name of activation row @p spec, or null for none. */
                const ActivationName* find_activation(const Config::LayerSpec& spec)
                {
                    for (const ActivationName& activation : K_ACTIVATIONS)
                    {
                        if (activation.type == spec.type)
                            return &activation;
                    }
                    return nullptr;
                }

                /** @brief Returns whether @p spec is an activation Darknet states as it is. */
                Result<bool> is_stated_activation(const Config::LayerSpec& spec)
                {
                    const RowKeys keys = row_keys(spec);
                    const ActivationName* activation = find_activation(spec);
                    if (activation == nullptr)
                        return false;
                    const bool leaky = activation->type == "leaky_relu";
                    const SushiRuntime::span<const std::string_view> allowed =
                        leaky ? SushiRuntime::span<const std::string_view>(
                                    K_LEAKY_KEYS, std::size(K_LEAKY_KEYS))
                              : SushiRuntime::span<const std::string_view>();
                    if (!reject_unknown_keys(keys, allowed).is_ok())
                        return false;
                    if (!leaky)
                        return true;
                    SA_TRY_ASSIGN(const double slope, read_real_key(keys, "slope", -1.0));
                    return slope == K_LEAKY_SLOPE;
                }

                /** @brief Returns Unsupported unless batchnorm2d row @p spec is Darknet's. */
                Status check_normalisation(const Config::LayerSpec& spec, std::size_t row)
                {
                    const RowKeys keys = row_keys(spec);
                    const Status checked = reject_unknown_keys(
                        keys, SushiRuntime::span<const std::string_view>(
                                  K_BATCHNORM_KEYS, std::size(K_BATCHNORM_KEYS)));
                    SA_FAIL_IF(!checked.is_ok(), StatusCode::Unsupported,
                               "its batchnorm2d row {} states a key other than bn_epsilon", row);
                    SA_TRY_ASSIGN(const double epsilon, read_real_key(keys, "bn_epsilon", 0.0));
                    SA_FAIL_IF(epsilon != K_DARKNET_EPSILON, StatusCode::Unsupported,
                               "its batchnorm2d row {} states bn_epsilon {}, and Darknet's is "
                               "1e-5",
                               row, epsilon);
                    return Status();
                }

                /** @brief Returns the float32 vector `<row>.<field>` of @p count entries. */
                Result<std::vector<float>> read_vector(const DarknetExportProgress& progress,
                                                       std::size_t row, std::string_view field,
                                                       std::int64_t count)
                {
                    std::vector<std::int64_t> extents;
                    SA_TRY_ASSIGN(std::vector<float> values,
                                  read_weight(*progress.model, row, field,
                                              K_DARKNET_FLOAT32_HOLDER, extents));
                    SA_FAIL_IF(static_cast<std::int64_t>(values.size()) != count,
                               StatusCode::InvalidDescription,
                               "the weight '{}.{}' holds {} values, and the row has {} filters",
                               row, field, values.size(), count);
                    return values;
                }

                /** @brief Returns the kernel of row @p row as [out, in, kh, kw]. */
                Result<std::vector<float>> read_kernel(const DarknetExportProgress& progress,
                                                       std::size_t row, const Convolution& read,
                                                       std::int64_t channels)
                {
                    std::vector<std::int64_t> record;
                    SA_TRY_ASSIGN(const std::vector<float> stored,
                                  read_weight(*progress.model, row, "weight",
                                              K_DARKNET_FLOAT32_HOLDER, record));
                    SA_FAIL_IF(record.size() != K_KERNEL_RANK || record[0] != read.filters ||
                                   record[1] != read.size || record[2] != read.size ||
                                   record[3] != channels,
                               StatusCode::InvalidDescription,
                               "the weight '{}.weight' is not [{}, {}, {}, {}]", row,
                               read.filters, read.size, read.size, channels);
                    std::vector<float> laid_out(stored.size());
                    IO::copy_kernel_first(stored.data(),
                                          IO::FilterExtents{record[0], record[3], record[1],
                                                            record[2]},
                                          laid_out.data());
                    return laid_out;
                }

                /** @brief Returns the section's lines for @p read and its activation. */
                std::vector<DarknetOptionText> describe_section(const Convolution& read,
                                                                bool normalised,
                                                                std::string_view activation)
                {
                    std::vector<DarknetOptionText> options;
                    if (normalised)
                        options.push_back({"batch_normalize", "1"});
                    options.push_back({"filters", spell_integer(read.filters)});
                    options.push_back({"size", spell_integer(read.size)});
                    options.push_back({"stride", spell_integer(read.stride)});
                    if (read.padding == read.size / 2)
                        options.push_back({"pad", "1"});
                    else
                        options.push_back({"padding", spell_integer(read.padding)});
                    options.push_back({"activation", std::string(activation)});
                    return options;
                }
            } // namespace

            Status write_darknet_conv2d(const DarknetRowCall& call,
                                        DarknetExportProgress& progress)
            {
                SA_TRY_ASSIGN(const Convolution read, read_convolution(*call.spec));
                SA_TRY(check_source_count(call, 1));
                const std::vector<std::int64_t>& input = call.values[0]->shape;
                const std::int64_t span_h = input[0] + 2 * read.padding - read.size;
                const std::int64_t span_w = input[1] + 2 * read.padding - read.size;
                SA_FAIL_IF(span_h < 0 || span_w < 0, StatusCode::InvalidDescription,
                           "row {} 'conv2d' does not fit its window in the input", call.index);

                std::size_t tail = call.index;
                const Config::LayerSpec* next = sole_follower(progress, tail);
                const bool normalised = next != nullptr && next->type == "batchnorm2d";
                if (normalised)
                {
                    SA_TRY(check_normalisation(*next, tail + 1));
                    ++tail;
                    next = sole_follower(progress, tail);
                }
                SA_FAIL_IF(normalised && read.use_bias, StatusCode::Unsupported,
                           "it adds a bias before a batch normalisation, and a normalised "
                           "Darknet convolution has none");
                SA_FAIL_IF(!normalised && !read.use_bias, StatusCode::Unsupported,
                           "it adds no bias and no batch normalisation follows it, and Darknet "
                           "gives such a convolution a bias");
                std::string_view activation = "linear";
                if (next != nullptr)
                {
                    SA_TRY_ASSIGN(const bool stated, is_stated_activation(*next));
                    if (stated)
                    {
                        activation = find_activation(*next)->name;
                        ++tail;
                    }
                }

                DarknetConvolutionFloats floats;
                if (!progress.model->weights.empty())
                {
                    SA_TRY_ASSIGN(floats.weights,
                                  read_kernel(progress, call.index, read, input[2]));
                    const std::size_t owner = normalised ? call.index + 1 : call.index;
                    SA_TRY_ASSIGN(floats.biases,
                                  read_vector(progress, owner, "bias", read.filters));
                    if (normalised)
                    {
                        SA_TRY_ASSIGN(floats.scales,
                                      read_vector(progress, owner, "weight", read.filters));
                        SA_TRY_ASSIGN(floats.means,
                                      read_vector(progress, owner, "running_mean", read.filters));
                        SA_TRY_ASSIGN(floats.variances,
                                      read_vector(progress, owner, "running_var", read.filters));
                    }
                }

                SA_TRY(follow_row(progress, call.sources[0]));
                add_section(progress, "convolutional",
                            describe_section(read, normalised, activation));
                progress.convolutions.push_back(std::move(floats));
                for (std::size_t row = call.index + 1; row <= tail; ++row)
                    progress.absorbed[row] = true;
                define_row_value(progress, tail,
                                 {span_h / read.stride + 1, span_w / read.stride + 1,
                                  read.filters});
                return Status();
            }
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
