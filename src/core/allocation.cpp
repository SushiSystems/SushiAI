/**************************************************************************/
/* allocation.cpp                                                         */
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

#include <SushiAI/core/allocation.hpp>

namespace SushiAI
{
    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine,
                                      std::span<const int64_t> dims,
                                      DType dtype)
    {
        int64_t elements = 1;
        for (int64_t extent : dims)
        {
            SA_THROW_IF(extent <= 0, "Tensor extents must be positive; got {}", extent);
            elements *= extent;
        }

        SA_THROW_IF(elements <= 0, "Cannot allocate a tensor with no elements");

        auto storage = SushiRuntime::make_sushi<SushiBLAS::Storage>(
            engine.get_context().get_allocator(),
            static_cast<std::size_t>(elements) * dtype_size(dtype),
            SushiRuntime::Memory::Strategies::Shared::instance());

        return SushiBLAS::Tensor(storage, dims, dtype, 0, engine.get_layout());
    }

    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine, const Shape& shape, DType dtype)
    {
        return allocate_tensor(engine,
                               std::span<const int64_t>(shape.data(),
                                                        static_cast<std::size_t>(shape.rank())),
                               dtype);
    }

    SushiBLAS::Tensor allocate_like(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& like)
    {
        SA_THROW_IF(like.num_elements <= 0, "Cannot allocate a tensor shaped like an empty one");

        // The layout comes from the source rather than the engine: a tensor
        // that was built column-major must be matched, or gemm would derive a
        // different leading dimension for the copy than for the original.
        int64_t elements = 1;
        for (int32_t i = 0; i < like.rank; ++i)
            elements *= like.shape[static_cast<std::size_t>(i)];

        auto storage = SushiRuntime::make_sushi<SushiBLAS::Storage>(
            engine.get_context().get_allocator(),
            static_cast<std::size_t>(elements) * dtype_size(like.dtype),
            SushiRuntime::Memory::Strategies::Shared::instance());

        return SushiBLAS::Tensor(storage,
                                 std::span<const int64_t>(like.shape.data(),
                                                          static_cast<std::size_t>(like.rank)),
                                 like.dtype, 0, like.layout);
    }

} // namespace SushiAI
