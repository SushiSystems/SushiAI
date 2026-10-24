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
            SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
            const SushiBLAS::PackLayout& layout, std::vector<SushiBLAS::Tensor> parameters)
        {
            SA_TRY(require_layout(layout, parameters));

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                SushiBLAS::TensorPack pack(engine.runtime(), layout);
                binding.share(pack);

                // The first replay's tasks read the pack before any gather has run.
                unsigned char* const base = static_cast<unsigned char*>(pack.data());
                const std::vector<SushiBLAS::PackEntry>& entries = layout.entries();
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    if (entries[index].size_bytes != 0)
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
