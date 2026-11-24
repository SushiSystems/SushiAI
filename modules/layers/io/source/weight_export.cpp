/**************************************************************************/
/* weight_export.cpp                                                      */
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
 * @file weight_export.cpp
 * @brief Implements the turn of a model's state records into the tensors of a weight file.
 * @author Mustafa Garip
 */

#include <SushiAI/io/weight_export.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/io/filter_layout.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/imported_model.hpp>
#include <SushiAI/io/name_map.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace IO
    {
        namespace
        {
            /** @brief Holds the suffix of the running mean a batch normalisation keeps. */
            constexpr std::string_view K_RUNNING_MEAN = "running_mean";

            /** @brief Returns the entry named @p name, or null when none is. */
            [[nodiscard]] const NN::StateEntry* find_entry(
                SushiRuntime::span<const NN::StateEntry> entries, std::string_view name) noexcept
            {
                for (const NN::StateEntry& entry : entries)
                {
                    if (entry.name == name)
                        return &entry;
                }
                return nullptr;
            }

            /** @brief Returns @p name with the first rule whose target it starts with undone. */
            [[nodiscard]] std::string file_name_of(const std::string& name,
                                                   const std::vector<RenameRule>& rules)
            {
                for (const RenameRule& rule : rules)
                {
                    if (name.compare(0, rule.to_prefix.size(), rule.to_prefix) == 0)
                        return std::string(rule.from_prefix) + name.substr(rule.to_prefix.size());
                }
                return name;
            }

            /** @brief Returns the filter extents of a kernel record of @p shape. */
            [[nodiscard]] FilterExtents extents_of(const Shape& shape, bool depthwise) noexcept
            {
                FilterExtents extents;
                if (depthwise)
                {
                    extents.outputs = shape[2];
                    extents.inputs = 1;
                    extents.height = shape[0];
                    extents.width = shape[1];
                    return extents;
                }
                extents.outputs = shape[0];
                extents.inputs = shape[3];
                extents.height = shape[1];
                extents.width = shape[2];
                return extents;
            }

            /** @brief Moves @p tensor, a kernel, to the layout of the file. */
            [[nodiscard]] Status move_kernel(ImportedTensor& tensor, bool depthwise)
            {
                SA_FAIL_IF(tensor.record.dtype != DType::Float32, StatusCode::Unsupported,
                           "the kernel '{}' is {}, and a kernel is moved as float32",
                           tensor.record.name, dtype_name(tensor.record.dtype));
                const FilterExtents extents = extents_of(tensor.record.shape, depthwise);
                const std::size_t count = tensor.payload.size() / sizeof(float);
                std::vector<float> source(count);
                std::vector<float> moved(count);
                std::memcpy(source.data(), tensor.payload.data(), tensor.payload.size());
                if (depthwise)
                    copy_depthwise_kernel_first(source.data(), extents, moved.data());
                else
                    copy_kernel_first(source.data(), extents, moved.data());
                std::memcpy(tensor.payload.data(), moved.data(), tensor.payload.size());

                tensor.record.shape = depthwise
                    ? Shape{extents.outputs, 1, extents.height, extents.width}
                    : Shape{extents.outputs, extents.inputs, extents.height, extents.width};
                return Status();
            }

            /** @brief Returns the reason @p tensor does not fit @p entry, or empty when it fits. */
            [[nodiscard]] std::string misfit(const ImportedTensor& tensor,
                                             const NN::StateEntry* entry)
            {
                if (entry == nullptr)
                    return "'" + tensor.record.name + "' has no entry in the model";
                if (entry->shape != tensor.record.shape)
                    return "'" + tensor.record.name + "' is " + tensor.record.shape.to_string() +
                           " and its entry " + entry->shape.to_string();
                const std::int32_t rank = entry->layout == NN::StateLayout::KernelLast ? 4
                                       : entry->layout == NN::StateLayout::DepthwiseKernelLast
                                           ? 3
                                           : entry->shape.rank();
                if (entry->shape.rank() != rank)
                    return "'" + tensor.record.name + "' is a kernel of rank " +
                           std::to_string(entry->shape.rank());
                return std::string();
            }
        } // namespace

        Result<ExportedWeights> export_state_weights(
            SushiRuntime::span<const ImportedTensor> records,
            SushiRuntime::span<const NN::StateEntry> entries, const NameMap& map) noexcept
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                [&]() -> Result<ExportedWeights>
            {
                std::string misfits;
                for (const ImportedTensor& record : records)
                {
                    const std::string reason =
                        misfit(record, find_entry(entries, record.record.name));
                    if (!reason.empty())
                        misfits.append(misfits.empty() ? "" : "; ").append(reason);
                }
                SA_FAIL_IF(!misfits.empty(), StatusCode::Incompatible,
                           "the weight file does not fit the model: {}", misfits);

                const std::vector<RenameRule> rules = map.rename_rules();
                ExportedWeights out;
                bool normalised = false;
                for (const ImportedTensor& record : records)
                {
                    const NN::StateEntry* entry = find_entry(entries, record.record.name);
                    ImportedTensor tensor = record;
                    if (entry->layout != NN::StateLayout::Plain)
                    {
                        SA_TRY(move_kernel(tensor,
                                           entry->layout == NN::StateLayout::DepthwiseKernelLast));
                    }
                    const std::string& name = record.record.name;
                    normalised = normalised ||
                                 (name.size() >= K_RUNNING_MEAN.size() &&
                                  name.compare(name.size() - K_RUNNING_MEAN.size(),
                                               K_RUNNING_MEAN.size(), K_RUNNING_MEAN) == 0);
                    tensor.record.name = file_name_of(name, rules);
                    out.tensors.push_back(std::move(tensor));
                }

                for (const std::string& drop : map.drop_names())
                    out.notes.push_back("the map drops '" + drop +
                                        "', which SushiAI has no record of, so it is not written");
                if (normalised)
                    out.notes.push_back("num_batches_tracked is not written: SushiAI keeps no "
                                        "count, so load the file with "
                                        "load_state_dict(strict=False)");
                return out;
            });
        }

    } // namespace IO
} // namespace SushiAI
