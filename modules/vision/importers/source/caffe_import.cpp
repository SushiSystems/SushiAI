/**************************************************************************/
/* caffe_import.cpp                                                       */
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
 * @file caffe_import.cpp
 * @brief Defines the Caffe importer: the layer walk, the published outputs and the file reads.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/caffe_import.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/caffe_net.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/protobuf_text.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/seam/seam.hpp>

#include "caffe_operator_table.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the files the importer takes at least: the prototxt. */
            constexpr std::size_t K_FEWEST_FILES = 1;

            /** @brief Holds the files the importer takes at most: the prototxt and the caffemodel. */
            constexpr std::size_t K_MOST_FILES = 2;

            /** @brief Returns whether @p message states the phase @p phase in a field @p key. */
            [[nodiscard]] bool states_phase(const Interchange::CaffeLayer& layer,
                                            std::string_view key, std::string_view phase)
            {
                for (const IO::TextField* rule : IO::find_fields(layer.parameters, key))
                {
                    const IO::TextField* stated = IO::find_field(*rule, "phase");
                    if (stated != nullptr && stated->text == phase)
                        return true;
                }
                return false;
            }

            /** @brief Returns whether @p layer exists only in the training net. */
            [[nodiscard]] bool is_train_only(const Interchange::CaffeLayer& layer)
            {
                return states_phase(layer, "include", "TRAIN") ||
                       states_phase(layer, "exclude", "TEST");
            }

            /**
             * @brief Returns Unsupported listing every layer of @p net whose type has no row.
             *
             * One run shows the whole gap.
             */
            [[nodiscard]] Status reject_unsupported_layers(const Interchange::CaffeNet& net,
                                                           std::string_view origin)
            {
                std::string listed;
                std::size_t count = 0;
                for (const Interchange::CaffeLayer& layer : net.layers)
                {
                    if (is_train_only(layer) || Detail::Caffe::find_operator_translator(layer.type) != nullptr)
                        continue;
                    listed += (count == 0 ? "" : ", ") + Detail::Caffe::describe_layer(layer);
                    ++count;
                }
                SA_FAIL_IF(count != 0, StatusCode::Unsupported,
                           "'{}': {} layers have no row in this importer: {}", origin, count,
                           listed);
                return Status();
            }

            /** @brief Declares the inputs the net states at its own level. */
            [[nodiscard]] Status declare_net_inputs(const Interchange::CaffeNet& net,
                                                    Detail::Caffe::TranslationProgress& progress)
            {
                SA_FAIL_IF(net.inputs.size() != net.input_shapes.size(),
                           StatusCode::InvalidDescription,
                           "the net names {} inputs and gives {} shapes", net.inputs.size(),
                           net.input_shapes.size());
                for (std::size_t i = 0; i < net.inputs.size(); ++i)
                {
                    SA_TRY(Detail::Caffe::declare_input(progress, net.inputs[i], net.input_shapes[i], -1));
                }
                return Status();
            }

            /** @brief Runs the translator of each layer of the net, in file order. */
            [[nodiscard]] Status translate_layers(const Interchange::CaffeNet& net,
                                                  Detail::Caffe::TranslationProgress& progress)
            {
                for (std::size_t at = 0; at < net.layers.size(); ++at)
                {
                    const Interchange::CaffeLayer& layer = net.layers[at];
                    if (progress.absorbed[at] || is_train_only(layer))
                        continue;
                    const Detail::Caffe::OperatorTableRow* row =
                        Detail::Caffe::find_operator_translator(layer.type);
                    SA_TRY(row->translate(layer, static_cast<std::int64_t>(at), progress));
                }
                return Status();
            }

            /** @brief Returns whether a layer that was translated reads @p name after @p writer. */
            [[nodiscard]] bool is_read_after(const Interchange::CaffeNet& net,
                                             const Detail::Caffe::TranslationProgress& progress,
                                             const std::string& name, std::int64_t writer)
            {
                for (std::size_t at = static_cast<std::size_t>(writer + 1); at < net.layers.size();
                     ++at)
                {
                    if (progress.absorbed[at] || is_train_only(net.layers[at]))
                        continue;
                    for (const std::string& bottom : net.layers[at].bottoms)
                    {
                        if (bottom == name)
                            return true;
                    }
                }
                return false;
            }

            /**
             * @brief Publishes every blob that no layer reads after the layer that wrote it.
             * @return InvalidDescription when the net has none.
             */
            [[nodiscard]] Status publish_outputs(const Interchange::CaffeNet& net,
                                                 Detail::Caffe::TranslationProgress& progress,
                                                 std::string_view origin)
            {
                for (const Detail::Caffe::NamedBlob& blob : progress.blobs)
                {
                    if (blob.value.row == Config::K_MODEL_INPUT ||
                        is_read_after(net, progress, blob.name, blob.writer))
                        continue;
                    progress.spec.outputs.push_back(Config::OutputSpec{
                        blob.value.row, std::string(NN::K_SINGLE_PORT_NAME), blob.name});
                }
                SA_FAIL_IF(progress.spec.outputs.empty(), StatusCode::InvalidDescription,
                           "'{}': no blob of the net is left unread, so it publishes nothing",
                           origin);
                return Status();
            }

            /** @brief Returns the import of @p net; the body of import_caffe_net. */
            [[nodiscard]] Result<IO::ImportedModel> translate_net(const Interchange::CaffeNet& net,
                                                                  std::string_view origin)
            {
                Detail::Caffe::TranslationProgress progress;
                progress.net = &net;
                progress.absorbed.assign(net.layers.size(), false);
                for (const Interchange::CaffeLayer& layer : net.layers)
                    progress.has_weights = progress.has_weights || !layer.blobs.empty();

                SA_TRY(reject_unsupported_layers(net, origin));
                SA_TRY(declare_net_inputs(net, progress));
                SA_TRY(translate_layers(net, progress));
                SA_FAIL_IF(!progress.has_input, StatusCode::InvalidDescription,
                           "'{}': the net declares no input", origin);
                SA_FAIL_IF(progress.spec.layers.empty(), StatusCode::InvalidDescription,
                           "'{}': the net has no layer that becomes a row", origin);
                SA_TRY(publish_outputs(net, progress, origin));

                IO::ImportedModel imported;
                imported.model = std::move(progress.spec);
                imported.weights = std::move(progress.tensors);
                return imported;
            }

            /** @brief Returns the text of the file at @p path. */
            [[nodiscard]] Result<std::string> read_text(const std::filesystem::path& path)
            {
                std::ifstream in(path, std::ios::binary);
                SA_FAIL_IF(!in, StatusCode::FileNotFound, "could not open '{}' for reading",
                           path.string());
                std::string text((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
                SA_FAIL_IF(in.bad(), StatusCode::InvalidFile, "could not read '{}'", path.string());
                return text;
            }

            /** @brief Returns the bytes of the file at @p path. */
            [[nodiscard]] Result<std::vector<std::byte>> read_bytes(const std::filesystem::path& path)
            {
                SA_TRY_ASSIGN(const std::string text, read_text(path));
                std::vector<std::byte> bytes(text.size());
                for (std::size_t i = 0; i < text.size(); ++i)
                    bytes[i] = static_cast<std::byte>(text[i]);
                return bytes;
            }

            /** @brief Returns the import of the files of @p files; the body of import_caffe_files. */
            [[nodiscard]] Result<IO::ImportedModel> read_files(
                SushiRuntime::span<const std::filesystem::path> files)
            {
                SA_FAIL_IF(files.size() < K_FEWEST_FILES || files.size() > K_MOST_FILES,
                           StatusCode::InvalidDescription,
                           "the '{}' format takes a .prototxt and optionally a .caffemodel, but {} "
                           "files were given",
                           K_CAFFE_FORMAT, files.size());

                SA_TRY_ASSIGN(const std::string text, read_text(files[0]));
                SA_TRY_ASSIGN(Interchange::CaffeNet net,
                              Interchange::read_caffe_prototxt(text, files[0].string()));
                if (files.size() == K_FEWEST_FILES)
                    return translate_net(net, files[0].string());

                // The blobs of the weights view these bytes, so they live until the import ends.
                SA_TRY_ASSIGN(const std::vector<std::byte> bytes, read_bytes(files[1]));
                SA_TRY_ASSIGN(const Interchange::CaffeNet weights,
                              Interchange::read_caffe_model(
                                  SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                                  files[1].string()));
                SA_TRY(Interchange::attach_caffe_weights(net, weights));
                return translate_net(net, files[0].string());
            }
        } // namespace

        Result<IO::ImportedModel> import_caffe_net(const Interchange::CaffeNet& net,
                                                   std::string_view origin) noexcept
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return translate_net(net, origin);
            });
        }

        Result<IO::ImportedModel> import_caffe_files(
            SushiRuntime::span<const std::filesystem::path> files) noexcept
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return read_files(files);
            });
        }
    } // namespace Vision::Importers
} // namespace SushiAI
