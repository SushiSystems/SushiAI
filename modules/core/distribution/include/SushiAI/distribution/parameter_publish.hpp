/**************************************************************************/
/* parameter_publish.hpp                                                  */
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
 * @file parameter_publish.hpp
 * @brief Declares ParameterPublish, which packs the parameters behind the optimizer.
 * @author Mustafa Garip
 * @see modules/core/distribution/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <SushiBLAS/cluster/cluster_binding.hpp>
#include <SushiBLAS/cluster/pack_layout.hpp>
#include <SushiBLAS/cluster/tensor_pack.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Distribution
    {
        /**
         * @brief Owns the parameter pack the replica tasks read and records the gather that
         *        writes it, behind the optimizer of a compiled step.
         *
         * The gather is a task on the master, so it raises the pack's version and the next
         * replay's replica tasks pull the new parameters.
         */
        class ParameterPublish final : public Graph::StepRecorder
        {
            public:
                /**
                 * @brief Allocates the pack of @p layout, shares it through @p binding, which
                 *        outlives the publish, and copies @p parameters into it on the host.
                 * @param parameters One contiguous, host-addressable tensor per entry, in order,
                 *        all on the device the pack is allocated on.
                 * @return The publish; InvalidDescription for a tensor count, or a tensor unlike
                 *         the layout or off the first's device; or a failed allocation or share.
                 */
                [[nodiscard]] static Result<std::unique_ptr<ParameterPublish>> make(
                    SushiBLAS::Engine& engine, SushiBLAS::ClusterBinding& binding,
                    const SushiBLAS::PackLayout& layout,
                    std::vector<SushiBLAS::Tensor> parameters) noexcept;

                ParameterPublish(const ParameterPublish&) = delete;
                ParameterPublish& operator=(const ParameterPublish&) = delete;
                ParameterPublish(ParameterPublish&&) = delete;
                ParameterPublish& operator=(ParameterPublish&&) = delete;
                ~ParameterPublish() override = default;

                /** @brief Returns the pack the replica tasks read, shared with the cluster. */
                [[nodiscard]] const SushiBLAS::TensorPack& pack() const noexcept
                {
                    return pack_;
                }

                /**
                 * @brief Records one task that gathers every parameter into the pack.
                 * @return 1, or the failure of the recording.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const Graph::StepRecording& recording) override;

            private:
                /** @brief Takes the shared pack and the parameters it gathers. */
                ParameterPublish(SushiBLAS::TensorPack pack,
                                 std::vector<SushiBLAS::Tensor> parameters) noexcept;

                /// @brief The parameter pack; copies name the same buffer.
                SushiBLAS::TensorPack pack_;

                /// @brief The parameters, in the order of the pack's layout.
                std::vector<SushiBLAS::Tensor> parameters_;
        };

    } // namespace Distribution
} // namespace SushiAI
