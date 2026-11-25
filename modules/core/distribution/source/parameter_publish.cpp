/**************************************************************************/
/* parameter_publish.cpp                                                  */
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
 * @file parameter_publish.cpp
 * @brief Implements ParameterPublish: the checks, the first copy and the recorded gather.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/parameter_publish.hpp>

#include <utility>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include "storage_device.hpp"

namespace SushiAI
{
    namespace Distribution
    {
        namespace
        {
            /**
             * @brief Checks that @p tensors match the entries of @p layout one for one, on one
             *        device.
             * @return That device, or InvalidDescription naming the first tensor that differs.
             */
            [[nodiscard]] Result<std::size_t> require_layout(
                const SushiBLAS::PackLayout& layout, const std::vector<SushiBLAS::Tensor>& tensors)
            {
                const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
                SA_FAIL_IF(tensors.size() != entries.size(), StatusCode::InvalidDescription,
                           "The parameter pack holds {} tensors and {} parameters were given",
                           entries.size(), tensors.size());

                std::size_t device = 0;
                for (std::size_t index = 0; index < tensors.size(); ++index)
                {
                    SA_FAIL_IF(!SushiBLAS::PackLayout::matches(entries[index], tensors[index]),
                               StatusCode::InvalidDescription,
                               "Parameter {} differs from its pack entry in type, layout or "
                               "shape",
                               index);
                    SA_FAIL_IF(!tensors[index].storage || !tensors[index].is_contiguous(),
                               StatusCode::InvalidDescription,
                               "Parameter {} has no storage or is not contiguous", index);
                    SA_TRY_ASSIGN(const std::size_t placed,
                                  require_pack_device(tensors[index], "Parameter", index));
                    if (index == 0)
                        device = placed;
                    SA_FAIL_IF(placed != device, StatusCode::InvalidDescription,
                               "Parameter {} lives on device {} and parameter 0 on device {}",
                               index, placed, device);
                }
                return device;
            }

            /**
             * @brief Copies each of @p tensors into its place in @p pack's pinned host bytes.
             * @return Success, or the failure of a copy.
             * @pre require_layout() accepted @p tensors for the pack's layout.
             */
            [[nodiscard]] Status copy_into(const SushiBLAS::TensorPack& pack,
                                           const std::vector<SushiBLAS::Tensor>& tensors)
            {
                unsigned char* const base = static_cast<unsigned char*>(pack.data());
                const std::vector<SushiBLAS::PackEntry>& entries = pack.layout().entries();
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    SA_TRY(copy_to_host(tensors[index], 0, tensors[index].num_elements,
                                        dtype_size(tensors[index].dtype),
                                        base + entries[index].offset_bytes));
                }
                return Status();
            }
        } // namespace

        ParameterPublish::ParameterPublish(SushiBLAS::TensorPack pack,
                                           std::vector<SushiBLAS::Tensor> parameters) noexcept
            : pack_(std::move(pack)), parameters_(std::move(parameters))
        {

        }

        Result<std::unique_ptr<ParameterPublish>> ParameterPublish::make(
            RuntimeAccess& runtime, SushiBLAS::ClusterBinding& binding,
            const SushiBLAS::PackLayout& layout, std::vector<SushiBLAS::Tensor> parameters) noexcept
        {
            SA_TRY_ASSIGN(const std::size_t device, require_layout(layout, parameters));

            SA_TRY_ASSIGN(SushiBLAS::TensorPack pack,
                          SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate,
                                                       [&]()
            {
                SushiBLAS::TensorPack shared(runtime.runtime(), layout, device);
                binding.share(shared);
                return shared;
            }));
            SA_TRY(copy_into(pack, parameters));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<ParameterPublish>(
                    new ParameterPublish(std::move(pack), std::move(parameters)));
            });
        }

        Result<std::size_t> ParameterPublish::record_into_step(
            const Graph::StepRecording& recording)
        {
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Record, [&]()
            {
                pack_.record_gather(recording.tasks(),
                                    SushiRuntime::span<const SushiBLAS::Tensor>(
                                        parameters_.data(), parameters_.size()));
                return Status();
            }));

            return std::size_t{1};
        }

    } // namespace Distribution
} // namespace SushiAI
