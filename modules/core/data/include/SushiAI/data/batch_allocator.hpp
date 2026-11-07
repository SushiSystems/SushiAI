/**************************************************************************/
/* batch_allocator.hpp                                                    */
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

/**
 * @file batch_allocator.hpp
 * @brief Declares BatchAllocator, the port a batch is allocated through, and its two adapters.
 * @author Mustafa Garip
 * @see modules/core/data/README.md
 */

#pragma once

#include <cstdint>
#include <SushiRuntime/core/span.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Data
    {
        /**
         * @class BatchAllocator
         * @brief Allocates the tensors of a batch.
         *
         * allocate_batch() and prepare_batch() take one, so the layout of a batch is decided
         * without a device when the adapter holds none.
         */
        class BatchAllocator
        {
            public:
                virtual ~BatchAllocator() = default;

                /**
                 * @brief Allocates a tensor of @p dims.
                 * @param dims The extents, outermost first.
                 * @param dtype The element type.
                 * @return The tensor, InvalidDescription for an extent that is not positive, or
                 *         the failure of the allocation.
                 */
                [[nodiscard]] virtual Result<SushiBLAS::Tensor> allocate(
                    SushiRuntime::span<const int64_t> dims, DType dtype) noexcept = 0;
        };

        /**
         * @class EngineBatchAllocator
         * @brief Allocates each tensor with its own Storage on an engine, as allocate_tensor()
         *        does.
         */
        class EngineBatchAllocator final : public BatchAllocator
        {
            public:
                /** @brief Draws from @p engine, which must outlive the allocator. */
                explicit EngineBatchAllocator(SushiBLAS::Engine& engine) noexcept : engine_(&engine)
                {

                }

                [[nodiscard]] Result<SushiBLAS::Tensor> allocate(
                    SushiRuntime::span<const int64_t> dims, DType dtype) noexcept override;

            private:
                /// @brief Holds the engine whose context owns the allocator.
                SushiBLAS::Engine* engine_;
        };

        /**
         * @class InMemoryBatchAllocator
         * @brief Describes each tensor and gives it no Storage; touches no device, file, thread
         *        or clock.
         *
         * A tensor it returns holds its extents, element type and layout and no element, so a
         * test reads the layout of a batch and cannot read a value.
         */
        class InMemoryBatchAllocator final : public BatchAllocator
        {
            public:
                [[nodiscard]] Result<SushiBLAS::Tensor> allocate(
                    SushiRuntime::span<const int64_t> dims, DType dtype) noexcept override;
        };

    } // namespace Data
} // namespace SushiAI
