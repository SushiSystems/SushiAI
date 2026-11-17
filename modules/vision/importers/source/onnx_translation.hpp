/**************************************************************************/
/* onnx_translation.hpp                                                   */
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
 * @file onnx_translation.hpp
 * @brief Declares the state of an ONNX translation and the helpers every operator shares.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/interchange/constant_folding.hpp>
#include <SushiAI/interchange/onnx_graph.hpp>
#include <SushiAI/io/imported_model.hpp>

namespace SushiAI
{
    namespace Vision::Importers
    {
        /** @brief Holds the units of the ONNX translation that no caller names. */
        namespace Detail
        {
            /** @brief Names what the fusion plan makes of a node. */
            enum class OnnxNodeRole : std::uint8_t
            {
                Plain, /**< The operator is translated as its own row. */
                Gate,  /**< A Sigmoid whose Mul becomes a silu row; it writes no row. */
                Silu,  /**< The Mul of a Sigmoid and the value the Sigmoid reads; one silu row. */
            };

            /** @brief Holds a computed tensor: the row that writes it and its shape. */
            struct OnnxValue
            {
                /** @brief Holds the row that writes the value, or Config::K_MODEL_INPUT. */
                std::int64_t row = Config::K_MODEL_INPUT;

                /** @brief Holds the extents of one sample, channel first: [C, H, W] or [F]. */
                std::vector<std::int64_t> dimensions;

                /** @brief Holds the [C, H, W] a vector was flattened from; empty otherwise. */
                std::vector<std::int64_t> flattened_from;
            };

            /** @brief Holds a constant tensor read as numbers: an initializer or a folded value. */
            struct OnnxConstant
            {
                std::vector<std::int64_t> dimensions; /**< The extents; empty for a scalar. */
                bool is_real = false;                 /**< Whether reals, not integers, holds it. */
                std::vector<float> reals;             /**< The elements of a floating tensor. */
                std::vector<std::int64_t> integers;   /**< The elements of an integer tensor. */

                /** @brief Returns the number of elements. */
                [[nodiscard]] std::size_t count() const noexcept
                {
                    return is_real ? reals.size() : integers.size();
                }
            };

            /** @brief Holds a translation under way: its rows, weights and names. */
            struct OnnxProgress
            {
                /** @brief Holds the graph being translated; its initializers are read in place. */
                const Interchange::OnnxGraph* graph = nullptr;

                /** @brief Holds the folded form of the graph's nodes and their constants. */
                const Interchange::FoldedGraph* folded = nullptr;

                /** @brief Holds every computed tensor defined so far, by name. */
                std::unordered_map<std::string, OnnxValue> values;

                /** @brief Holds the constants read or made so far, by name. */
                std::unordered_map<std::string, OnnxConstant> constants;

                /** @brief Holds the rows of the model file. */
                std::vector<Config::Value> layers;

                /** @brief Holds the weights, named by the row that owns them. */
                std::vector<IO::ImportedTensor> weights;

                /** @brief Holds the role the fusion plan gave the node being translated. */
                OnnxNodeRole role = OnnxNodeRole::Plain;

                /** @brief Holds, for a Silu node, the name of the tensor its Sigmoid reads. */
                std::string gate_source;
            };

            /** @brief Holds the window of a convolution or a pooling along both image axes. */
            struct OnnxWindow
            {
                std::int64_t kernel[2] = {1, 1};    /**< Height and width of the window. */
                std::int64_t stride[2] = {1, 1};    /**< Step along height and width. */
                std::int64_t dilation[2] = {1, 1};  /**< Spacing of the taps. */
                std::int64_t pad_begin[2] = {0, 0}; /**< Zero rows above and columns left. */
                std::int64_t pad_end[2] = {0, 0};   /**< Zero rows below and columns right. */
            };

            /** @brief Returns "node 'name' (Op)", or "node (Op)" for an unnamed node. */
            [[nodiscard]] std::string describe_node(const Interchange::OnnxNode& node);

            /** @brief Returns whether operand @p position of @p node is stated and not empty. */
            [[nodiscard]] bool has_operand(const Interchange::OnnxNode& node,
                                           std::size_t position) noexcept;

            /** @brief Returns InvalidDescription unless @p node has @p fewest to @p most inputs. */
            [[nodiscard]] Status check_operand_count(const Interchange::OnnxNode& node,
                                                     std::size_t fewest, std::size_t most);

            /** @brief Returns Unsupported unless @p node has exactly @p count outputs. */
            [[nodiscard]] Status check_output_count(const Interchange::OnnxNode& node,
                                                    std::size_t count);

            /** @brief Returns Unsupported naming the first attribute of @p node not in @p known. */
            [[nodiscard]] Status reject_unknown_attributes(
                const Interchange::OnnxNode& node,
                SushiRuntime::span<const std::string_view> known);

            /** @brief Returns the integer attribute @p name, or @p fallback when it is absent. */
            [[nodiscard]] Result<std::int64_t> read_int_attribute(const Interchange::OnnxNode& node,
                                                                  std::string_view name,
                                                                  std::int64_t fallback);

            /** @brief Returns the integer list attribute @p name; empty when it is absent. */
            [[nodiscard]] Result<std::vector<std::int64_t>> read_ints_attribute(
                const Interchange::OnnxNode& node, std::string_view name);

            /** @brief Returns the float attribute @p name, or @p fallback when it is absent. */
            [[nodiscard]] Result<float> read_real_attribute(const Interchange::OnnxNode& node,
                                                            std::string_view name, float fallback);

            /** @brief Returns the string attribute @p name, or @p fallback when it is absent. */
            [[nodiscard]] Result<std::string> read_text_attribute(const Interchange::OnnxNode& node,
                                                                  std::string_view name,
                                                                  std::string_view fallback);

            /**
             * @brief Returns the computed tensor named @p name.
             * @return InvalidDescription when no node or input defines it; Unsupported when it
             *         is a constant, which a row cannot read as a value.
             */
            [[nodiscard]] Result<const OnnxValue*> find_value(OnnxProgress& progress,
                                                              const Interchange::OnnxNode& node,
                                                              const std::string& name);

            /** @brief Returns the computed tensor operand @p position of @p node reads. */
            [[nodiscard]] Result<const OnnxValue*> find_operand_value(
                OnnxProgress& progress, const Interchange::OnnxNode& node, std::size_t position);

            /**
             * @brief Returns the constant tensor operand @p position of @p node reads.
             * @return Unsupported when the operand is computed at run time, or an integer,
             *         float32 or float64 tensor is not what the operand holds.
             */
            [[nodiscard]] Result<const OnnxConstant*> find_operand_constant(
                OnnxProgress& progress, const Interchange::OnnxNode& node, std::size_t position);

            /** @brief Returns @p tensor as numbers; Unsupported for a type other than the four. */
            [[nodiscard]] Result<OnnxConstant> read_constant_tensor(
                const Interchange::OnnxTensor& tensor);

            /** @brief Records @p constant as the tensor @p name, for later operands to read. */
            void define_constant(OnnxProgress& progress, const std::string& name,
                                 OnnxConstant constant);

            /** @brief Records the value @p name as written by @p row with @p dimensions. */
            void define_value(OnnxProgress& progress, const std::string& name, std::int64_t row,
                              std::vector<std::int64_t> dimensions,
                              std::vector<std::int64_t> flattened_from = {});

            /**
             * @brief Appends a row of @p type reading @p sources and returns its index.
             * @param keys The attribute keys of the row; @p values holds one value for each.
             * @param sources The rows it reads, in order; Config::K_MODEL_INPUT is the input.
             */
            [[nodiscard]] std::int64_t append_row(OnnxProgress& progress, std::string_view type,
                                                  std::vector<std::string> keys,
                                                  std::vector<Config::Value> values,
                                                  const std::vector<std::int64_t>& sources);

            /** @brief Appends a row of @p type with no attribute over @p value; returns its row. */
            [[nodiscard]] std::int64_t append_pointwise_row(OnnxProgress& progress,
                                                            std::string_view type,
                                                            const OnnxValue& value);

            /**
             * @brief Translates a node that combines two computed tensors of one shape into one
             *        row of @p row_type, which reads them in operand order.
             * @return InvalidDescription for a count of operands other than two; Unsupported for
             *         an operand that is a constant, an output count other than one and two
             *         tensors of different extents, since the rows do not broadcast across the
             *         channel-first extents.
             */
            [[nodiscard]] Status translate_binary_row(const Interchange::OnnxNode& node,
                                                      OnnxProgress& progress,
                                                      std::string_view row_type);

            /**
             * @brief Appends a flatten row over @p input and defines @p output as its vector.
             * @pre @p input is an image of rank 3.
             */
            void append_flatten(OnnxProgress& progress, const std::string& output,
                                const OnnxValue& input);

            /**
             * @brief Appends a linear row over @p input and defines the output of @p node.
             * @param weight The weight as ONNX stores it for Y = X W^T: [out, in], row-major.
             * @param bias The bias [out], or null for a row with none.
             * @return InvalidDescription when the input is no vector of `in` features.
             * @note A vector flattened from an image is in the (H, W, C) order of the flatten
             *       row, so the columns of the weight move from (C, H, W) to (H, W, C).
             */
            [[nodiscard]] Status append_linear(OnnxProgress& progress,
                                               const Interchange::OnnxNode& node,
                                               const OnnxValue& input,
                                               const std::vector<std::int64_t>& weight_extents,
                                               const std::vector<float>& weight,
                                               const std::vector<float>* bias);

            /**
             * @brief Stores @p values as the float32 weight `<row>.<field>` of extents @p shape.
             * @return InvalidDescription when @p shape is no shape or its product is not the
             *         number of @p values.
             */
            [[nodiscard]] Status store_weight(OnnxProgress& progress, std::int64_t row,
                                              std::string_view field,
                                              const std::vector<std::int64_t>& shape,
                                              const std::vector<float>& values);

            /**
             * @brief Returns operand @p position of @p node as a float constant of @p rank.
             * @return Unsupported when it is not a floating constant; InvalidDescription when
             *         its rank differs.
             */
            [[nodiscard]] Result<const OnnxConstant*> find_real_operand(
                OnnxProgress& progress, const Interchange::OnnxNode& node, std::size_t position,
                std::size_t rank);

            /** @brief Returns the integer when @p height equals @p width, else the array [h, w]. */
            [[nodiscard]] Config::Value pair_value(std::int64_t height, std::int64_t width);

            /**
             * @brief Reads kernel_shape, strides, dilations and pads of @p node into a window.
             * @param kernel What kernel_shape must equal, [h, w] of the weight; empty when the
             *        node states it alone.
             * @return Unsupported for a rank other than 2, auto_pad other than NOTSET and
             *         negative pads; InvalidDescription for a stated kernel that disagrees.
             */
            [[nodiscard]] Result<OnnxWindow> read_window(const Interchange::OnnxNode& node,
                                                         const std::vector<std::int64_t>& kernel);

            /**
             * @brief Returns the extent a window leaves along one axis of @p input.
             * @return InvalidDescription naming @p node when the window does not fit.
             */
            [[nodiscard]] Result<std::int64_t> window_extent(const Interchange::OnnxNode& node,
                                                             const OnnxWindow& window,
                                                             std::size_t axis,
                                                             std::int64_t input);

            /** @brief Returns the opset of the graph's default domain, 0 when it states none. */
            [[nodiscard]] std::int64_t default_opset(const OnnxProgress& progress) noexcept;
        } // namespace Detail
    } // namespace Vision::Importers
} // namespace SushiAI
