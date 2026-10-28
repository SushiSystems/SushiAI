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
 * @file parameter_publish.cpp
 * @brief Implements ParameterPublish: the checks, the first copy and the recorded gather.
 * @author Mustafa Garip
 */

#include <SushiAI/distribution/parameter_publish.hpp>

#include <cstring>
#include <utility>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        namespace
        {
            /**
             * @brief Checks that @p tensors match the entries of @p layout one for one.
             * @return Success, or InvalidDescription naming the first tensor that differs.
             */
            [[nodiscard]] Status require_layout(const SushiBLAS::PackLayout& layout,
                                                const std::vector<SushiBLAS::Tensor>& tensors)
            {
                const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
                SA_FAIL_IF(tensors.size() != entries.size(), StatusCode::InvalidDescription,
                           "The parameter pack holds {} tensors and {} parameters were given",
                           entries.size(), tensors.size());

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
                    SA_FAIL_IF(tensors[index].storage->device_index()
                                   != tensors.front().storage->device_index(),
                               StatusCode::InvalidDescription,
                               "Parameter {} lives on device {} and parameter 0 on device {}",
                               index, tensors[index].storage->device_index(),
                               tensors.front().storage->device_index());
                }
                return Status();
            }

            /**
             * @brief Returns the device the tensors of a checked layout live on.
             * @return The device of the first tensor, or 0, the engine's default, for none.
             * @pre require_layout() accepted @p tensors.
             */
            [[nodiscard]] std::size_t device_of(const std::vector<SushiBLAS::Tensor>& tensors)
                noexcept
            {
                return tensors.empty() ? 0 : tensors.front().storage->device_index();
            }
        } // namespace

        ParameterPublish::ParameterPublish(SushiBLAS::TensorPack pack,
                                           std::vector<SushiBLAS::Tensor> parameters) noexcept
            : pack_(std::move(pack)), parameters_(std::move(parameters))
        {

        }

        Result<std::unique_ptr<ParameterPublish>> ParameterPublish::make(
            SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
            const SushiBLAS::PackLayout& layout, std::vector<SushiBLAS::Tensor> parameters) noexcept
        {
            SA_TRY(require_layout(layout, parameters));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                SushiBLAS::TensorPack pack(engine.runtime(), layout, device_of(parameters));
                binding.share(pack);

                // A replay reads the pack before a gather; its Shared storage takes the host copy.
                unsigned char* const base = static_cast<unsigned char*>(pack.data());
                const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    if (entries[index].size_bytes == 0)
                        continue;
                    require_host_visible(parameters[index], "Published parameter", index);
                    std::memcpy(base + entries[index].offset_bytes, parameters[index].data(),
                                entries[index].size_bytes);
                }

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
