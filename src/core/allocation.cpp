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

#include <exception>
#include <limits>
#include <string>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/allocation.hpp>

namespace SushiAI
{
    namespace
    {
        /**
         * @brief The element count of @p dims, refusing anything that would wrap.
         *
         * The obvious @c elements *= extent is signed overflow - undefined
         * behaviour, which a compiler is entitled to assume cannot happen and
         * therefore to delete the @c elements <= 0 check that was meant to
         * catch it. A shape whose product does not fit is a caller mistake, so
         * it is detected before the multiply rather than inferred from its
         * wreckage.
         *
         * @param dims The extents.
         * @return The product of the extents.
         * @throws Error If an extent is not positive or the product overflows.
         */
        int64_t checked_element_count(SushiRuntime::span<const int64_t> dims)
        {
            int64_t elements = 1;

            for (int64_t extent : dims)
            {
                SA_THROW_IF(extent <= 0, "Tensor extents must be positive; got {}", extent);
                SA_THROW_IF(extent > std::numeric_limits<int64_t>::max() / elements,
                            "A tensor of these extents holds more than {} elements, which no "
                            "index arithmetic in this library can address; the extent that "
                            "overflowed it is {}",
                            std::numeric_limits<int64_t>::max(), extent);
                elements *= extent;
            }

            return elements;
        }

        /**
         * @brief Renders extents as "[a, b]", for a diagnostic.
         * @param dims The extents.
         * @return The bracketed rendering.
         */
        std::string describe(SushiRuntime::span<const int64_t> dims)
        {
            std::string text = "[";
            for (std::size_t i = 0; i < dims.size(); ++i)
            {
                if (i) text += ", ";
                text += std::to_string(dims[i]);
            }
            return text + "]";
        }

        /**
         * @brief Builds a Storage of @p bytes, naming the tensor if it cannot.
         *
         * SushiBLAS already refuses a failed allocation loudly rather than
         * handing back a null pointer for a kernel to dereference - but its
         * message knows only a byte count. The shape and element type that
         * asked for those bytes live here, and a caller looking at "allocation
         * failed for 4398046511104 bytes" cannot work out which of a model's
         * hundred tensors it was. So the context is attached where it exists.
         *
         * @param engine The engine whose context owns the allocator.
         * @param bytes How many bytes to allocate.
         * @param dims The extents, for the diagnostic.
         * @param dtype The element type, for the diagnostic.
         * @return The new storage.
         * @throws Error If the underlying allocation fails.
         */
        SushiRuntime::sushi_ptr<SushiBLAS::Storage> make_storage(SushiBLAS::Engine& engine,
                                                                 std::size_t bytes,
                                                                 SushiRuntime::span<const int64_t> dims,
                                                                 DType dtype)
        {
            try
            {
                return SushiRuntime::make_sushi<SushiBLAS::Storage>(
                    engine.get_context().get_allocator(),
                    bytes,
                    SushiRuntime::Memory::Strategies::Shared::instance());
            }
            catch (const std::exception& failure)
            {
                throw Error(SushiBLAS::Core::format(
                    "Could not allocate the {} tensor {} ({} bytes): {}",
                    dtype_name(dtype), describe(dims), bytes, failure.what()));
            }
        }
    }

    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine,
                                      SushiRuntime::span<const int64_t> dims,
                                      DType dtype)
    {
        // No "did that come out as zero" check: checked_element_count already
        // refuses a non-positive extent and an overflowing product, so the only
        // remaining way to be unrepresentable is in bytes rather than elements.
        const int64_t elements = checked_element_count(dims);

        SA_THROW_IF(static_cast<std::uint64_t>(elements)
                        > std::numeric_limits<std::size_t>::max() / dtype_size(dtype),
                    "A {} tensor of {} elements needs more bytes than this platform can "
                    "express in a size_t", dtype_name(dtype), elements);

        auto storage = make_storage(engine, static_cast<std::size_t>(elements) * dtype_size(dtype),
                                    dims, dtype);

        return SushiBLAS::Tensor(storage, dims, dtype, 0, engine.get_layout());
    }

    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine, const Shape& shape, DType dtype)
    {
        return allocate_tensor(engine,
                               SushiRuntime::span<const int64_t>(shape.data(),
                                                                 static_cast<std::size_t>(shape.rank())),
                               dtype);
    }

    SushiBLAS::Tensor allocate_like(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& like)
    {
        SA_THROW_IF(like.num_elements <= 0, "Cannot allocate a tensor shaped like an empty one");

        // The layout comes from the source rather than the engine: a tensor
        // that was built column-major must be matched, or gemm would derive a
        // different leading dimension for the copy than for the original.
        const SushiRuntime::span<const int64_t> dims(like.shape.data(),
                                                     static_cast<std::size_t>(like.rank));
        const int64_t elements = checked_element_count(dims);

        auto storage = make_storage(engine,
                                    static_cast<std::size_t>(elements) * dtype_size(like.dtype),
                                    dims, like.dtype);

        return SushiBLAS::Tensor(storage, dims, like.dtype, 0, like.layout);
    }

    SushiRuntime::sushi_ptr<SushiBLAS::Storage> allocate_arena(SushiBLAS::Engine& engine,
                                                               std::size_t bytes)
    {
        // SushiBLAS::Storage refuses a zero-byte request, and it is right to:
        // there is no valid pointer to hand back. An empty arena is a planner
        // bug rather than a degenerate case to accommodate, so it is named here
        // instead of surfacing as "Requested size cannot be zero".
        SA_THROW_IF(bytes == 0,
                    "An arena of zero bytes cannot be allocated; a plan that packs nothing "
                    "should ask for no arena at all");

        try
        {
            return SushiRuntime::make_sushi<SushiBLAS::Storage>(
                engine.get_context().get_allocator(),
                bytes,
                SushiRuntime::Memory::Strategies::Shared::instance());
        }
        catch (const std::exception& failure)
        {
            throw Error(SushiBLAS::Core::format("Could not allocate a {} byte arena: {}",
                                                bytes, failure.what()));
        }
    }

    SushiBLAS::Tensor view_tensor(SushiBLAS::Engine& engine,
                                  const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
                                  std::size_t byte_offset,
                                  const Shape& shape,
                                  DType dtype)
    {
        SA_THROW_IF(!arena, "Cannot view into a null arena");

        const std::size_t element = dtype_size(dtype);

        SA_THROW_IF(byte_offset % element != 0,
                    "A {} view cannot start at byte {} of an arena: Tensor::storage_offset "
                    "counts elements, so a start that is not a multiple of the {}-byte element "
                    "width has no representation. The planner must round each tensor's start up "
                    "to its own element width.",
                    dtype_name(dtype), byte_offset, element);

        const std::size_t bytes = static_cast<std::size_t>(shape.num_elements()) * element;

        SA_THROW_IF(byte_offset + bytes > arena->size_bytes,
                    "A {} {} view at byte {} needs {} bytes and would run past the end of a "
                    "{} byte arena",
                    dtype_name(dtype), shape.to_string(), byte_offset, bytes, arena->size_bytes);

        return SushiBLAS::Tensor(arena,
                                 SushiRuntime::span<const int64_t>(shape.data(),
                                                                   static_cast<std::size_t>(shape.rank())),
                                 dtype,
                                 static_cast<int64_t>(byte_offset / element),
                                 engine.get_layout());
    }

} // namespace SushiAI
