/****************************************************************/
/* allocation.cpp                                               */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file allocation.cpp
 * @brief Implements tensor allocation and arena viewing operations.
 * @author Mustafa Garip
 */

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
         * @brief Returns element count of @p dims, verifying non-overflow.
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
         * @brief Allocates Storage of @p bytes with diagnostic context.
         * @param engine Engine context owner; @p bytes Byte count.
         * @param dims Extents; @p dtype Element type for diagnostics.
         * @return The new storage instance.
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
                    engine.get_context(),
                    bytes,
                    SushiRuntime::API::Residency::Shared);
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
        // checked_element_count validates positive extents and non-overflowing product.
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

        // Layout matches the source tensor to preserve matrix dimensions.
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
        // Arena requests must allocate at least one byte.
        SA_THROW_IF(bytes == 0,
                    "An arena of zero bytes cannot be allocated; a plan that packs nothing "
                    "should ask for no arena at all");

        try
        {
            return SushiRuntime::make_sushi<SushiBLAS::Storage>(
                engine.get_context(),
                bytes,
                SushiRuntime::API::Residency::Shared);
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
