/**************************************************************************/
/* trailing_chain.hpp                                                     */
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
 * @file trailing_chain.hpp
 * @brief Declares TrailingChain, which runs a step's copies, update and blends as one recorder.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/trailing_spec.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/graph/buffer_update_recorder.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Records the work of a TrailingSpec against the tensors of a StorageBank.
         *
         * The chain holds the tensors by shared storage, so it stays valid while the bank keeps
         * every id of the spec.
         */
        class TrailingChain final : public Graph::StepRecorder
        {
            public:
                /**
                 * @brief Resolves every id of @p spec in @p bank.
                 * @pre The bank holds every id of the spec.
                 * @return The chain, or OutOfMemory when the host allocation fails.
                 */
                [[nodiscard]] static Result<std::unique_ptr<TrailingChain>> make(
                    const TrailingSpec& spec, const StorageBank& bank) noexcept;

                /**
                 * @brief Records the copies, then the caller's recorder, then the blends.
                 * @return One task per copy plus what the others recorded, or the first failure.
                 */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const Graph::StepRecording& recording) override;

                /** @brief Refreshes the caller's recorder before a replay. */
                void before_replay() override;

                /** @brief Tells the caller's recorder that a replay finished. */
                void after_replay() override;

            private:
                /** @brief Holds one copy as the two tensors it touches. */
                struct CopySlot
                {
                    SushiBLAS::Tensor from; /**< The tensor the regions write. */
                    SushiBLAS::Tensor to;   /**< The tensor that receives the bytes. */
                };

                /** @brief Takes the resolved parts. */
                TrailingChain(Graph::StepRecorder* inner, std::vector<CopySlot> copies,
                              std::vector<Graph::BufferUpdateSlot> blends) noexcept;

                /// @brief The caller's recorder, or null; not owned.
                Graph::StepRecorder* inner_;

                /// @brief The gradient copies, in the order they run.
                std::vector<CopySlot> copies_;

                /// @brief Blends the running statistics.
                Graph::BufferUpdateRecorder blends_;
        };

    } // namespace Eager
} // namespace SushiAI
