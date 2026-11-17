/**************************************************************************/
/* onnx_import.cpp                                                        */
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
 * @file onnx_import.cpp
 * @brief Defines the ONNX importer over the graph reader, the operator table and the plan.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/importers/onnx_import.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/constant_folding.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/imported_model.hpp>
#include "onnx_fusion.hpp"
#include "onnx_operator_table.hpp"
#include "onnx_translation.hpp"

namespace SushiAI
{
    namespace Vision::Importers
    {
        namespace
        {
            /** @brief Holds the first opset whose operand forms the translators read. */
            constexpr std::int64_t K_FIRST_OPSET = 13;

            /** @brief Holds the rank of the graph input: batch, channels, height and width. */
            constexpr std::size_t K_INPUT_RANK = 4;

            /** @brief Holds the name of the one port every layer row publishes. */
            constexpr std::string_view K_PORT = "output";

            /** @brief Holds the shape and name of the one image a graph reads. */
            struct GraphInput
            {
                std::string name;        /**< The input's name. */
                std::int64_t channels;   /**< The channels C. */
                std::int64_t height;     /**< The rows H. */
                std::int64_t width;      /**< The columns W. */
            };

            /** @brief Returns the one input of @p graph that is not an initializer. */
            Result<GraphInput> read_graph_input(const Interchange::OnnxGraph& graph,
                                                std::string_view origin)
            {
                const Interchange::OnnxValueInfo* found = nullptr;
                for (const Interchange::OnnxValueInfo& input : graph.inputs)
                {
                    if (graph.find_initializer(input.name) != nullptr)
                        continue;
                    SA_FAIL_IF(found != nullptr, StatusCode::Unsupported,
                               "{}: the graph reads more than one input, '{}' and '{}'", origin,
                               found->name, input.name);
                    found = &input;
                }
                SA_FAIL_IF(found == nullptr, StatusCode::InvalidDescription,
                           "{}: the graph has no input", origin);
                SA_FAIL_IF(found->element_type != Interchange::OnnxDataType::Float,
                           StatusCode::Unsupported,
                           "{}: the input '{}' is not float32, and only a float32 image is read",
                           origin, found->name);
                SA_FAIL_IF(!found->has_shape || found->dimensions.size() != K_INPUT_RANK,
                           StatusCode::Unsupported,
                           "{}: the input '{}' does not state a shape [N, C, H, W]", origin,
                           found->name);
                for (std::size_t axis = 1; axis < K_INPUT_RANK; ++axis)
                {
                    SA_FAIL_IF(found->dimensions[axis].value < 1, StatusCode::Unsupported,
                               "{}: axis {} of the input '{}' is not fixed ({}), and the model "
                               "file states one input shape",
                               origin, axis, found->name,
                               found->dimensions[axis].parameter.empty()
                                   ? std::string("unstated")
                                   : found->dimensions[axis].parameter);
                }
                return GraphInput{found->name, found->dimensions[1].value,
                                  found->dimensions[2].value, found->dimensions[3].value};
            }

            /** @brief Holds the nodes no row can say and the names they leave undefined. */
            struct Refusals
            {
                std::vector<std::string> reasons;         /**< One entry per refused node. */
                std::unordered_set<std::string> poisoned; /**< The outputs of the refused nodes. */
            };

            /** @brief Returns whether @p node reads a tensor a refused node left undefined. */
            bool reads_poisoned(const Interchange::OnnxNode& node, const Refusals& refusals)
            {
                for (const std::string& name : node.inputs)
                {
                    if (refusals.poisoned.count(name) != 0)
                        return true;
                }
                return false;
            }

            /** @brief Records that @p node is refused for @p reason. */
            void refuse(const Interchange::OnnxNode& node, std::string reason, Refusals& refusals)
            {
                refusals.reasons.push_back(std::move(reason));
                for (const std::string& name : node.outputs)
                    refusals.poisoned.insert(name);
            }

            /** @brief Returns the operator set domain a node names as "ai.onnx" or empty. */
            bool is_default_domain(const Interchange::OnnxNode& node) noexcept
            {
                return node.domain.empty() || node.domain == "ai.onnx";
            }

            /**
             * @brief Translates every node, collecting the ones no row can say.
             * @return A failure other than Unsupported at once; otherwise success, with the
             *         refusals in @p refusals.
             */
            Status translate_nodes(const Interchange::FoldedGraph& folded,
                                   const Detail::OnnxNodePlan& plan,
                                   Detail::OnnxProgress& progress, Refusals& refusals)
            {
                for (std::size_t at = 0; at < folded.nodes.size(); ++at)
                {
                    const Interchange::OnnxNode& node = folded.nodes[at];
                    progress.role = plan.roles[at];
                    progress.gate_source = plan.gate_sources[at];

                    const Detail::OnnxOperatorRow* row = Detail::find_onnx_operator(node.op_type);
                    if (!is_default_domain(node) || row == nullptr)
                    {
                        refuse(node,
                               Detail::describe_node(node) +
                                   (is_default_domain(node)
                                        ? " has an operator no translator reads"
                                        : " belongs to the domain '" + node.domain + "'"),
                               refusals);
                        continue;
                    }
                    if (reads_poisoned(node, refusals))
                    {
                        for (const std::string& name : node.outputs)
                            refusals.poisoned.insert(name);
                        continue;
                    }

                    const Status translated = row->translate(node, progress);
                    if (translated.is_ok())
                        continue;
                    SA_FAIL_IF(translated.code() != StatusCode::Unsupported, translated.code(),
                               "{}", translated.message());
                    refuse(node, std::string(translated.message()), refusals);
                }
                return Status();
            }

            /** @brief Returns the entry of the model file that publishes one graph output. */
            Result<Config::Value> write_output(const Detail::OnnxProgress& progress,
                                               const Interchange::OnnxValueInfo& output,
                                               std::string_view origin)
            {
                const auto found = progress.values.find(output.name);
                SA_FAIL_IF(found == progress.values.end(), StatusCode::Unsupported,
                           "{}: the output '{}' is a constant or is never written", origin,
                           output.name);
                SA_FAIL_IF(found->second.row == Config::K_MODEL_INPUT, StatusCode::Unsupported,
                           "{}: the output '{}' is the input itself, which no row writes", origin,
                           output.name);
                return Config::Value::object(
                    {"node", "port", "name"},
                    {Config::Value::integer(found->second.row),
                     Config::Value::string(std::string(K_PORT)),
                     Config::Value::string(output.name)});
            }

            /** @brief Returns the notes of the import: what the three files do not say. */
            std::vector<std::string> write_notes(const Interchange::FoldedGraph& folded,
                                                 const Detail::OnnxProgress& progress,
                                                 const Interchange::OnnxGraph& graph)
            {
                std::vector<std::string> notes;
                if (folded.folded_count != 0)
                    notes.push_back(std::to_string(folded.folded_count) +
                                    " nodes of shape arithmetic were folded into constants");
                notes.emplace_back("the model reads and publishes images channel-last (H, W, C), "
                                   "where ONNX is channel-first");
                for (const Interchange::OnnxValueInfo& output : graph.outputs)
                {
                    const auto found = progress.values.find(output.name);
                    if (found != progress.values.end() && !found->second.flattened_from.empty())
                        notes.push_back("the output '" + output.name +
                                        "' is flattened from an image in the order (H, W, C), and "
                                        "ONNX lists those features as (C, H, W)");
                }
                notes.emplace_back("an ONNX file states no task and no training recipe; "
                                   "`--task` attaches a task section");
                return notes;
            }

            /** @brief Reads the whole of the file at @p path. */
            Result<std::vector<std::byte>> read_file(const std::filesystem::path& path)
            {
                std::ifstream stream(path, std::ios::binary | std::ios::ate);
                SA_FAIL_IF(!stream.is_open(), StatusCode::FileNotFound,
                           "cannot open the ONNX file '{}'", path.string());
                const std::streamsize size = stream.tellg();
                SA_FAIL_IF(size <= 0, StatusCode::InvalidFile, "the ONNX file '{}' is empty",
                           path.string());
                stream.seekg(0);
                std::vector<std::byte> bytes(static_cast<std::size_t>(size));
                stream.read(reinterpret_cast<char*>(bytes.data()), size);
                SA_FAIL_IF(!stream, StatusCode::InvalidFile, "cannot read the ONNX file '{}'",
                           path.string());
                return bytes;
            }
        } // namespace

        Result<IO::ImportedModel> import_onnx_model(const Interchange::OnnxGraph& graph,
                                                    std::string_view origin) noexcept
        {
            const std::int64_t opset = graph.opset_version("");
            SA_FAIL_IF(opset < K_FIRST_OPSET, StatusCode::Unsupported,
                       "{}: the graph is of opset {}, and the translation reads opset {} or newer",
                       origin, opset, K_FIRST_OPSET);
            SA_TRY_ASSIGN(const GraphInput input, read_graph_input(graph, origin));
            SA_TRY_ASSIGN(const Interchange::FoldedGraph folded,
                          Interchange::fold_constants(graph));

            Detail::OnnxProgress progress;
            progress.graph = &graph;
            progress.folded = &folded;
            Detail::define_value(progress, input.name, Config::K_MODEL_INPUT,
                                 {input.channels, input.height, input.width});
            const Detail::OnnxNodePlan plan = Detail::plan_onnx_nodes(folded.nodes, graph.outputs);

            Refusals refusals;
            SA_TRY(SA_ANNOTATE(translate_nodes(folded, plan, progress, refusals), "{}", origin));
            std::string listed;
            for (const std::string& reason : refusals.reasons)
                listed.append(listed.empty() ? "" : "; ").append(reason);
            SA_FAIL_IF(!refusals.reasons.empty(), StatusCode::Unsupported,
                       "{}: {} nodes cannot be imported: {}", origin, refusals.reasons.size(),
                       listed);
            SA_FAIL_IF(graph.outputs.empty(), StatusCode::InvalidDescription,
                       "{}: the graph has no output", origin);

            std::vector<Config::Value> outputs;
            for (const Interchange::OnnxValueInfo& output : graph.outputs)
            {
                SA_TRY_ASSIGN(Config::Value entry, write_output(progress, output, origin));
                outputs.push_back(std::move(entry));
            }
            SA_FAIL_IF(progress.layers.empty(), StatusCode::Unsupported,
                       "{}: the graph holds no node that becomes a row", origin);

            const Config::Value shape = Config::Value::object(
                {"shape"}, {Config::Value::array({Config::Value::integer(input.height),
                                                  Config::Value::integer(input.width),
                                                  Config::Value::integer(input.channels)})});
            const Config::Value document = Config::Value::object(
                {"format_version", "input", "layers", "outputs"},
                {Config::Value::integer(Config::K_MODEL_VERSION), shape,
                 Config::Value::array(std::move(progress.layers)),
                 Config::Value::array(std::move(outputs))});
            SA_TRY_ASSIGN(Config::ModelSpec model, Config::parse_model(document, origin));

            IO::ImportedModel imported;
            imported.model = std::move(model);
            imported.weights = std::move(progress.weights);
            imported.notes = write_notes(folded, progress, graph);
            return imported;
        }

        Result<IO::ImportedModel> import_onnx_files(
            SushiRuntime::span<const std::filesystem::path> files) noexcept
        {
            SA_FAIL_IF(files.size() != 1, StatusCode::InvalidDescription,
                       "the '{}' format takes one .onnx file, but {} files were given",
                       K_ONNX_FORMAT, files.size());
            SA_TRY_ASSIGN(const std::vector<std::byte> bytes, read_file(files[0]));
            const std::string origin = files[0].string();
            SA_TRY_ASSIGN(const Interchange::OnnxGraph graph,
                          Interchange::read_onnx_graph(
                              SushiRuntime::span<const std::byte>(bytes.data(), bytes.size()),
                              origin));
            return import_onnx_model(graph, origin);
        }
    } // namespace Vision::Importers
} // namespace SushiAI
