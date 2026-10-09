/**************************************************************************/
/* allocation.hpp                                                         */
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
 * @file allocation.hpp
 * @brief Declares tensor allocation and arena viewing functions.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    /**
     * @brief Allocates a tensor of @p dims with its own Storage.
     * @param engine The engine whose context owns the allocator.
     * @param dims The extents, outermost first.
     * @param dtype The element type.
     * @return The new tensor, or InvalidDescription for an extent that is not positive or a
     *         size that overflows, or OutOfMemory when the allocator refuses the storage.
     */
    [[nodiscard]] Result<SushiBLAS::Tensor> allocate_tensor(SushiBLAS::Engine& engine,
                                                            SushiRuntime::span<const int64_t> dims,
                                                            DType dtype = DType::Float32);

    /**
     * @brief Allocates a tensor of @p shape with its own Storage.
     * @param engine The engine whose context owns the allocator.
     * @param shape The extents.
     * @param dtype The element type.
     * @return The new tensor, or the failure the span overload returns; nothing else references
     *         its Storage.
     */
    [[nodiscard]] Result<SushiBLAS::Tensor> allocate_tensor(SushiBLAS::Engine& engine,
                                                            const Shape& shape,
                                                            DType dtype = DType::Float32);

    /**
     * @brief Allocates a tensor with @p like's extents, dtype, and layout.
     * @param engine The engine whose context owns the allocator.
     * @param like The tensor to copy the description from.
     * @pre @p like holds at least one element; an empty one stops the process.
     * @return The new tensor, which shares nothing with @p like, or the failure the span
     *         overload returns.
     */
    [[nodiscard]] Result<SushiBLAS::Tensor> allocate_like(SushiBLAS::Engine& engine,
                                                          const SushiBLAS::Tensor& like);

    /**
     * @brief Allocates one arena buffer for memory planning.
     * @param engine The engine whose context owns the allocator.
     * @param bytes Number of bytes the arena holds; zero stops the process.
     * @return The newly allocated storage, or OutOfMemory when the allocator refuses it.
     */
    [[nodiscard]] Result<SushiRuntime::sushi_ptr<SushiBLAS::Storage>> allocate_arena(
        SushiBLAS::Engine& engine,
        std::size_t bytes);

    /**
     * @brief Returns a tensor view into @p arena at @p byte_offset.
     * @param engine The engine whose default layout the view takes.
     * @param arena Storage to carve from; must outlive the view.
     * @param byte_offset Offset in bytes; must be aligned to element width.
     * @return The view. Stops the process when the offset is misaligned or out of bounds.
     */
    [[nodiscard]] SushiBLAS::Tensor view_tensor(
        SushiBLAS::Engine& engine,
        const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
        std::size_t byte_offset,
        const Shape& shape,
        DType dtype);

} // namespace SushiAI
