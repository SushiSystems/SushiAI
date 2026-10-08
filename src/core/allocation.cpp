/**************************************************************************/
/* allocation.cpp                                                         */
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
 * @file allocation.cpp
 * @brief Implements tensor allocation and arena viewing operations.
 * @author Mustafa Garip
 */

#include <SushiAI/core/allocation.hpp>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace
    {
        /**
         * @brief Returns the element count of @p dims, verifying non-overflow.
         * @param dims The extents.
         * @return The product of the extents, or InvalidDescription when an extent is not
         *         positive or the product overflows.
         */
        [[nodiscard]] Result<int64_t> checked_element_count(SushiRuntime::span<const int64_t> dims)
        {
            int64_t elements = 1;

            for (int64_t extent : dims)
            {
                SA_FAIL_IF(extent <= 0, StatusCode::InvalidDescription,
                           "Tensor extents must be positive; got {}", extent);
                SA_FAIL_IF(extent > std::numeric_limits<int64_t>::max() / elements,
                           StatusCode::InvalidDescription,
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
         * @brief Allocates Storage of @p bytes, shared between host and device.
         * @param engine Engine whose context owns the allocator.
         * @param bytes Byte count.
         * @return The new storage, or the Status of what the allocator threw.
         */
        [[nodiscard]] Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>> reserve_storage(
            SushiBLAS::Engine& engine,
            std::size_t bytes)
        {
            return Detail::guard_value(Detail::SeamPhase::Allocate, [&engine, bytes]()
            {
                return SushiRuntime::make_sushi<SushiBLAS::Storage>(
                    engine.get_context(),
                    bytes,
                    SushiRuntime::API::Residency::Shared);
            });
        }

        /**
         * @brief Allocates Storage of @p bytes for a tensor, with diagnostic context.
         * @param engine Engine whose context owns the allocator.
         * @param bytes Byte count.
         * @param dims Extents, named in the failure's message with @p dtype.
         * @return The new storage, or OutOfMemory annotated with the tensor's description.
         */
        [[nodiscard]] Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>> make_storage(
            SushiBLAS::Engine& engine,
            std::size_t bytes,
            SushiRuntime::span<const int64_t> dims,
            DType dtype)
        {
            auto outcome = reserve_storage(engine, bytes);
            if (!outcome.has_value())
                return SA_ANNOTATE(outcome.status(),
                                   "Could not allocate the {} tensor {} ({} bytes)",
                                   dtype_name(dtype), describe(dims), bytes);

            return std::move(outcome).value();
        }
    }

    Result<SushiBLAS::Tensor> allocate_tensor(SushiBLAS::Engine& engine,
                                              SushiRuntime::span<const int64_t> dims,
                                              DType dtype)
    {
        // checked_element_count validates positive extents and non-overflowing product.
        SA_TRY_ASSIGN(const int64_t elements, checked_element_count(dims));

        SA_FAIL_IF(static_cast<std::uint64_t>(elements)
                       > std::numeric_limits<std::size_t>::max() / dtype_size(dtype),
                   StatusCode::InvalidDescription,
                   "A {} tensor of {} elements needs more bytes than this platform can "
                   "express in a size_t", dtype_name(dtype), elements);

        SA_TRY_ASSIGN(auto storage,
                      make_storage(engine, static_cast<std::size_t>(elements) * dtype_size(dtype),
                                   dims, dtype));

        return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
        {
            return SushiBLAS::Tensor(std::move(storage), dims, dtype, 0, engine.get_layout());
        });
    }

    Result<SushiBLAS::Tensor> allocate_tensor(SushiBLAS::Engine& engine,
                                              const Shape& shape,
                                              DType dtype)
    {
        const SushiRuntime::span<const int64_t> dims(shape.data(),
                                                     static_cast<std::size_t>(shape.rank()));
        return allocate_tensor(engine, dims, dtype);
    }

    Result<SushiBLAS::Tensor> allocate_like(SushiBLAS::Engine& engine,
                                            const SushiBLAS::Tensor& like)
    {
        SA_FATAL_IF(like.num_elements <= 0, "Cannot allocate a tensor shaped like an empty one");

        // The new tensor takes the layout of like, not the engine's.
        const SushiRuntime::span<const int64_t> dims(like.shape.data(),
                                                     static_cast<std::size_t>(like.rank));
        SA_TRY_ASSIGN(const int64_t elements, checked_element_count(dims));

        const std::size_t bytes = static_cast<std::size_t>(elements) * dtype_size(like.dtype);

        SA_TRY_ASSIGN(auto storage, make_storage(engine, bytes, dims, like.dtype));

        return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
        {
            return SushiBLAS::Tensor(std::move(storage), dims, like.dtype, 0, like.layout);
        });
    }

    Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>> allocate_arena(SushiBLAS::Engine& engine,
                                                                       std::size_t bytes)
    {
        // Arena requests must allocate at least one byte.
        SA_FATAL_IF(bytes == 0,
                    "An arena of zero bytes cannot be allocated; a plan that packs nothing "
                    "should ask for no arena at all");

        auto outcome = reserve_storage(engine, bytes);
        if (!outcome.has_value())
            return SA_ANNOTATE(outcome.status(), "Could not allocate a {} byte arena", bytes);

        return std::move(outcome).value();
    }

    SushiBLAS::Tensor view_tensor(SushiBLAS::Engine& engine,
                                  const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
                                  std::size_t byte_offset,
                                  const Shape& shape,
                                  DType dtype)
    {
        SA_FATAL_IF(!arena, "Cannot view into a null arena");

        const std::size_t element = dtype_size(dtype);

        SA_FATAL_IF(byte_offset % element != 0,
                    "A {} view cannot start at byte {} of an arena: Tensor::storage_offset "
                    "counts elements, so a start that is not a multiple of the {}-byte element "
                    "width has no representation. The planner must round each tensor's start up "
                    "to its own element width.",
                    dtype_name(dtype), byte_offset, element);

        const std::size_t bytes = static_cast<std::size_t>(shape.num_elements()) * element;

        SA_FATAL_IF(byte_offset + bytes > arena->size_bytes(),
                    "A {} {} view at byte {} needs {} bytes and would run past the end of a "
                    "{} byte arena",
                    dtype_name(dtype), shape.to_string(), byte_offset, bytes, arena->size_bytes());

        const SushiRuntime::span<const int64_t> dims(shape.data(),
                                                     static_cast<std::size_t>(shape.rank()));

        // A view the Tensor constructor refuses is fatal: value() stops the process on a failure.
        return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
        {
            return SushiBLAS::Tensor(arena,
                                     dims,
                                     dtype,
                                     static_cast<int64_t>(byte_offset / element),
                                     engine.get_layout());
        }).value();
    }

} // namespace SushiAI
