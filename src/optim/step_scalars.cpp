/****************************************************************/
/* step_scalars.cpp                                             */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
