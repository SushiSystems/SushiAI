/**************************************************************************/
/* step_scalars.cpp                                                       */
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
 * @file step_scalars.cpp
 * @brief Implements device-resident scalar buffer management for optimizer kernels.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/step_scalars.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Optim
    {
        Result<StepScalars> StepScalars::make(SushiBLAS::Engine& engine, std::size_t count)
        {
            SA_FAIL_IF(count == 0, StatusCode::InvalidDescription,
                       "A step-scalar buffer needs at least one slot");

            SA_TRY_ASSIGN(SushiBLAS::Tensor buffer,
                          allocate_tensor(engine, Shape{static_cast<int64_t>(count)},
                                          DType::Float32));

            // The buffer is shared device memory; zero it from the host before first use.
            std::fill_n(buffer.data_as<float>(), count, 0.0f);
            return StepScalars(std::move(buffer), count);
        }

        StepScalars::StepScalars(SushiBLAS::Tensor buffer, std::size_t count) noexcept
            : buffer_(std::move(buffer)), data_(buffer_.data_as<float>()), count_(count)
        {

        }

        void StepScalars::write(SushiRuntime::span<const float> values)
        {
            SA_FATAL_IF(values.size() != count_,
                        "This step-scalar buffer holds {} slots but {} values were written",
                        count_, values.size());

            for (std::size_t i = 0; i < count_; ++i) data_[i] = values[i];
        }

    } // namespace Optim
} // namespace SushiAI
