/**************************************************************************/
/* step_scalars.cpp                                                       */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file step_scalars.cpp
 * @brief Implements device-resident scalar buffer management for optimizer kernels.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/optim/step_scalars.hpp>

namespace SushiAI
{
    namespace Optim
    {
        StepScalars::StepScalars(SushiBLAS::Engine& engine, std::size_t count)
            : count_(count)
        {
            SA_THROW_IF(count == 0, "A step-scalar buffer needs at least one slot");

            buffer_ = allocate_tensor(engine, Shape{static_cast<int64_t>(count)},
                                      DType::FLOAT32);
            data_ = buffer_.data_as<float>();

            // Zero initialize shared USM buffer via host write before execution.
            for (std::size_t i = 0; i < count_; ++i) data_[i] = 0.0f;
        }

        void StepScalars::write(SushiRuntime::span<const float> values)
        {
            SA_THROW_IF(values.size() != count_,
                        "This step-scalar buffer holds {} slots but {} values were written",
                        count_, values.size());

            for (std::size_t i = 0; i < count_; ++i) data_[i] = values[i];
        }

    } // namespace Optim
} // namespace SushiAI
