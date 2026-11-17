/**************************************************************************/
/* weight_import.cpp                                                      */
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
 * @file weight_import.cpp
 * @brief Defines the strict match of a weight source against the state records of a model.
 * @author Mustafa Garip
 */

#include <SushiAI/io/weight_import.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/filter_layout.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/io/weight_source.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Marks a record no tensor of the file supplies. */
            constexpr std::size_t K_UNSUPPLIED = static_cast<std::size_t>(-1);

            /** @brief Appends "  <heading> (<count>): <items joined by ", ">" when items exist. */
            void append_section(std::string& message, std::string_view heading,
                                const std::vector<std::string>& items)
            {
                if (items.empty())
                    return;

                message.append("\n  ").append(heading).append(" (");
                message.append(std::to_string(items.size())).append("): ");
                for (std::size_t i = 0; i < items.size(); ++i)
                {
                    if (i != 0)
                        message.append(", ");
                    message.append(items[i]);
                }
            }

            /** @brief Returns @p name in quotes. */
            std::string in_quotes(std::string_view name)
            {
                return "'" + std::string(name) + "'";
            }

            /**
             * @brief Returns whether @p file is a convolution weight stored [out, in, kh, kw]
             *        that @p record states as [out, kh, kw, in].
             */
            bool is_channel_first(const Shape& file, const Shape& record) noexcept
            {
                return file.rank() == 4 && record.rank() == 4 && file[0] == record[0] &&
                       file[1] == record[3] && file[2] == record[1] && file[3] == record[2];
            }

            /**
             * @brief Returns whether @p file is a depthwise convolution weight stored
             *        [C, 1, kh, kw] that @p record states as [kh, kw, C].
             */
            bool is_depthwise_channel_first(const Shape& file, const Shape& record) noexcept
            {
                return file.rank() == 4 && record.rank() == 3 && file[1] == 1 &&
                       file[0] == record[2] && file[2] == record[0] && file[3] == record[1];
            }

            /** @brief Returns whether a tensor of shape @p file fills @p record, permuted or not. */
            bool fits(const Shape& file, const Shape& record) noexcept
            {
                return file == record || is_channel_first(file, record) ||
                       is_depthwise_channel_first(file, record);
            }

            /** @brief Holds what the file and the model disagree on, grouped by kind. */
            struct Mismatches
            {
                std::vector<std::string> missing;    /**< Records the file does not supply. */
                std::vector<std::string> extra;      /**< Tensors no record takes. */
                std::vector<std::string> misshapen; /**< Tensors whose shape differs. */

                /** @brief Returns whether the file and the model agree. */
                bool is_empty() const noexcept
                {
                    return missing.empty() && extra.empty() && misshapen.empty();
                }
            };

            /**
             * @brief Pairs every tensor of @p source with a record, or lists why it has none.
             * @param supplier Receives, for each record, the index of the tensor that supplies it.
             */
            Mismatches match(SushiRuntime::span<const NN::StateEntry> records,
                             const WeightSource& source, const NameMap& map,
                             std::vector<std::size_t>& supplier)
            {
                const std::vector<RenameRule> rules = map.rename_rules();
                std::unordered_map<std::string_view, std::size_t> position;
                for (std::size_t i = 0; i < records.size(); ++i)
                    position.emplace(records[i].name, i);

                supplier.assign(records.size(), K_UNSUPPLIED);
                Mismatches found;
                const SushiRuntime::span<const SourceTensor> tensors = source.tensors();
                for (std::size_t j = 0; j < tensors.size(); ++j)
                {
                    const SourceTensor& tensor = tensors[j];
                    if (map.is_dropped(tensor.name))
                        continue;

                    const std::string target = renamed(tensor.name, rules);
                    const std::string as =
                        target == tensor.name ? std::string() : " as " + in_quotes(target);
                    const auto at = position.find(target);
                    if (at == position.end())
                    {
                        found.extra.push_back(in_quotes(tensor.name) + as);
                        continue;
                    }
                    if (supplier[at->second] != K_UNSUPPLIED)
                    {
                        found.extra.push_back(in_quotes(tensor.name) + as + ", which " +
                                              in_quotes(tensors[supplier[at->second]].name) +
                                              " already supplies");
                        continue;
                    }

                    supplier[at->second] = j;
                    if (!fits(tensor.shape, records[at->second].shape))
                    {
                        found.misshapen.push_back(
                            in_quotes(target) + " is " + records[at->second].shape.to_string() +
                            " in the model and " + tensor.shape.to_string() + " in the file" +
                            (as.empty() ? std::string() : " (" + in_quotes(tensor.name) + ")"));
                    }
                }

                for (std::size_t i = 0; i < records.size(); ++i)
                {
                    if (supplier[i] == K_UNSUPPLIED)
                        found.missing.push_back(in_quotes(records[i].name));
                }
                return found;
            }

            /** @brief Returns the one message that names every mismatch of an import. */
            std::string describe(const Mismatches& found, std::string_view origin)
            {
                std::string message =
                    "importing '" + std::string(origin) + "' does not fit the model:";
                append_section(message, "records the file does not supply", found.missing);
                append_section(message, "tensors of the file that no record takes", found.extra);
                append_section(message, "tensors of another shape than their record",
                               found.misshapen);
                return message;
            }

            /** @brief Returns the bytes of @p values as @p dtype elements, host byte order. */
            std::vector<std::byte> payload_of(const std::vector<float>& values, DType dtype)
            {
                std::vector<std::byte> payload;
                if (dtype == DType::Float32)
                {
                    payload.resize(values.size() * sizeof(float));
                    if (!values.empty())
                        std::memcpy(payload.data(), values.data(), payload.size());
                    return payload;
                }

                payload.resize(values.size() * sizeof(double));
                for (std::size_t i = 0; i < values.size(); ++i)
                {
                    const double widened = static_cast<double>(values[i]);
                    std::memcpy(payload.data() + i * sizeof(double), &widened, sizeof(double));
                }
                return payload;
            }
        } // namespace

        Result<std::vector<ImportedTensor>> import_state_weights(
            SushiRuntime::span<const NN::StateEntry> records, const WeightSource& source,
            const NameMap& map) noexcept
        {
            std::vector<std::size_t> supplier;
            const Mismatches found = match(records, source, map, supplier);
            SA_FAIL_IF(!found.is_empty(), StatusCode::Incompatible, "{}",
                       describe(found, source.origin()));

            std::vector<ImportedTensor> out;
            out.reserve(records.size());
            for (std::size_t i = 0; i < records.size(); ++i)
            {
                const NN::StateEntry& record = records[i];
                SA_FAIL_IF(record.dtype != DType::Float32 && record.dtype != DType::Float64,
                           StatusCode::Unsupported,
                           "the record '{}' is {}, and a weight import fills float32 and float64 "
                           "records only",
                           record.name, dtype_name(record.dtype));

                const SourceTensor& tensor = source.tensors()[supplier[i]];
                SA_TRY_ASSIGN(std::vector<float> values, source.read_float32(supplier[i]));
                SA_FAIL_IF(static_cast<std::int64_t>(values.size()) != record.shape.num_elements(),
                           StatusCode::InvalidFile,
                           "'{}': the tensor '{}' reads {} elements, and its shape {} holds {}",
                           source.origin(), tensor.name, values.size(), tensor.shape.to_string(),
                           record.shape.num_elements());

                if (is_channel_first(tensor.shape, record.shape))
                {
                    std::vector<float> permuted(values.size());
                    const FilterExtents extents{tensor.shape[0], tensor.shape[1], tensor.shape[2],
                                                tensor.shape[3]};
                    copy_kernel_last(values.data(), extents, permuted.data());
                    values = std::move(permuted);
                }
                else if (is_depthwise_channel_first(tensor.shape, record.shape))
                {
                    std::vector<float> permuted(values.size());
                    const FilterExtents extents{tensor.shape[0], tensor.shape[1], tensor.shape[2],
                                                tensor.shape[3]};
                    copy_depthwise_kernel_last(values.data(), extents, permuted.data());
                    values = std::move(permuted);
                }

                ImportedTensor imported;
                imported.record = TensorRecord{record.name, record.dtype, record.shape};
                imported.payload = payload_of(values, record.dtype);
                out.push_back(std::move(imported));
            }
            return out;
        }

    } // namespace IO
} // namespace SushiAI
