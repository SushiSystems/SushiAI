/**************************************************************************/
/* step_scalars.hpp                                                       */
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
 * @file step_scalars.hpp
 * @brief Declares device-resident scalar buffer management for optimizer kernels.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>

namespace SushiAI
{
    namespace Optim
    {
        /**
         * @class StepScalars
         * @brief A tiny device-resident float vector, rewritten from the host per step.
         *
         * Non-copyable: the whole point of the object is the stability of the
         * one address its kernels captured, and a copy would suggest there can
         * be two.
         */
        class StepScalars final
        {
            public:
                /**
                 * @brief Allocates and zeroes the device buffer for step scalars.
                 * @param engine Engine whose context owns the allocator.
                 * @param count Number of float scalars the buffer holds; must be positive.
                 * @return The buffer, or InvalidDescription for a zero count, or the failure
                 *         allocate_tensor returns.
                 */
                [[nodiscard]] static Result<StepScalars> make(SushiBLAS::Engine& engine,
                                                              std::size_t count);

                StepScalars(const StepScalars&) = delete;
                StepScalars& operator=(const StepScalars&) = delete;
                StepScalars(StepScalars&&) noexcept = default;
                StepScalars& operator=(StepScalars&&) noexcept = default;
                ~StepScalars() = default;

                /**
                 * @brief The address a device kernel captures by value.
                 *
                 * Stable for the life of this object, which is what makes the
                 * capture legal across replays.
                 *
                 * @return A pointer to element 0 of the buffer.
                 */
                const float* device_pointer() const noexcept { return data_; }

                /**
                 * @brief Overwrites buffer contents from the host before replay.
                 * @pre @p values holds exactly count() elements; a mismatch stops the process.
                 */
                void write(SushiRuntime::span<const float> values);

                /** @brief How many scalars the buffer holds. */
                std::size_t count() const noexcept { return count_; }

            private:
                /**
                 * @brief Takes ownership of an allocated buffer.
                 * @param buffer Tensor of @p count float elements, owned from here on.
                 * @param count Number of elements in @p buffer.
                 */
                StepScalars(SushiBLAS::Tensor buffer, std::size_t count) noexcept;

                /// @brief The tensor that owns the device storage.
                SushiBLAS::Tensor buffer_;

                /// @brief Element 0 of buffer_, cached so that device_pointer() is free.
                float* data_ = nullptr;

                /// @brief Number of elements in buffer_.
                std::size_t count_ = 0;
        };

    } // namespace Optim
} // namespace SushiAI
