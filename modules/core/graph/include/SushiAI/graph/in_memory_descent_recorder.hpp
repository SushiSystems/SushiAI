/**************************************************************************/
/* in_memory_descent_recorder.hpp                                         */
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
 * @file in_memory_descent_recorder.hpp
 * @brief Declares InMemoryDescentRecorder, a StepRecorder that records one axpy per parameter.
 * @author Mustafa Garip
 * @see modules/core/graph/README.md
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class InMemoryDescentRecorder
         * @brief Records parameter -= rate * gradient for each pair it holds, as one axpy task
         *        per pair, and counts the calls it receives; opens no file and starts no thread.
         *
         * The tasks are real: they run on whatever engine the recording belongs to. It owns no
         * optimizer state, so it is the plainest update a step can carry.
         */
        class InMemoryDescentRecorder final : public StepRecorder
        {
            public:
                /**
                 * @brief Starts with no pair.
                 * @param rate The step size every pair is moved by.
                 */
                explicit InMemoryDescentRecorder(float rate) noexcept : rate_(rate)
                {
                }

                /** @brief Adds one parameter and the tensor its gradient is written into. */
                void add(const SushiBLAS::Tensor& parameter, const SushiBLAS::Tensor& gradient)
                {
                    pairs_.emplace_back(parameter, gradient);
                }

                /** @brief Records one axpy per pair and returns how many it recorded. */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const StepRecording& recording) override
                {
                    ++records_;
                    return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
                    {
                        SushiBLAS::BLASOps blas(recording.tasks());
                        for (std::pair<SushiBLAS::Tensor, SushiBLAS::Tensor>& pair : pairs_)
                            blas.axpy(-rate_, pair.second, pair.first);

                        return pairs_.size();
                    });
                }

                /** @brief Counts the runs that have finished. */
                void after_replay() override
                {
                    ++replays_;
                }

                /** @brief Returns how many times the update was recorded. */
                [[nodiscard]] std::size_t record_count() const noexcept
                {
                    return records_;
                }

                /** @brief Returns how many runs called after_replay(). */
                [[nodiscard]] std::size_t replay_count() const noexcept
                {
                    return replays_;
                }

            private:
                /// @brief Holds the step size.
                float rate_;

                /// @brief Holds each parameter with its gradient.
                std::vector<std::pair<SushiBLAS::Tensor, SushiBLAS::Tensor>> pairs_;

                /// @brief Counts the calls of record_into_step().
                std::size_t records_ = 0;

                /// @brief Counts the calls of after_replay().
                std::size_t replays_ = 0;
        };

    } // namespace Graph
} // namespace SushiAI
