/**************************************************************************/
/* in_memory_late_increment_recorder.hpp                                  */
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
 * @file in_memory_late_increment_recorder.hpp
 * @brief Declares InMemoryLateIncrementRecorder, a StepRecorder that records one late-count task.
 * @author Mustafa Garip
 * @see tests/support/README.md
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <SushiBLAS/engine/operand_set.hpp>
#include <SushiBLAS/engine/task_description.hpp>
#include <SushiBLAS/tensor/tensor.hpp>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @class InMemoryLateIncrementRecorder
         * @brief Records one late-count task that adds one to as many leading elements of a
         *        tensor as a host count holds; opens no file and starts no thread.
         *
         * The task reads the count when it runs, not when it is recorded, and captures the data
         * pointers only, so it replays after the recorder that made it is gone.
         */
        class InMemoryLateIncrementRecorder final : public Graph::StepRecorder
        {
            public:
                /**
                 * @brief Binds the update to the tensor it adds to and the count of its width.
                 * @param out The FLOAT32 tensor whose leading elements are incremented.
                 * @param count A one-element INT32 tensor in host memory.
                 */
                InMemoryLateIncrementRecorder(const SushiBLAS::Tensor& out,
                                              const SushiBLAS::Tensor& count)
                    : out_(out), count_(count)
                {
                }

                /** @brief Records one late-count task over the tensor and returns 1. */
                [[nodiscard]] Result<std::size_t> record_into_step(
                    const Graph::StepRecording& recording) override
                {
                    return ::SushiAI::Detail::guard_value(::SushiAI::Detail::SeamPhase::Record, [&]()
                    {
                        SushiBLAS::OperandSet operands;
                        operands.reads(count_);
                        operands.writes(out_);
                        float* const data = out_.data_as<float>();
                        const std::int32_t* const width = count_.data_as<std::int32_t>();
                        recording.tasks().record_late_pair(
                            SushiBLAS::TaskDescription(), operands, "late_increment", count_,
                            static_cast<std::size_t>(out_.num_elements),
                            [data, width]()
                            {
                                for (std::int32_t i = 0; i < width[0]; ++i)
                                    data[i] += 1.0f;
                            },
                            [data](std::size_t i)
                            {
                                data[i] += 1.0f;
                            });
                        return std::size_t{1};
                    });
                }

            private:
                /// @brief Holds the tensor each replay adds to.
                SushiBLAS::Tensor out_;

                /// @brief Holds the count of elements each replay adds to.
                SushiBLAS::Tensor count_;
        };

    } // namespace Test
} // namespace SushiAI
