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
#include <SushiBLAS/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    /**
     * @brief Allocates a tensor of @p dims with its own Storage.
     * @param engine The engine whose context owns the allocator.
     * @param dims The extents, outermost first.
     * @param dtype The element type.
     * @return The new tensor with standalone storage.
     * @throws Error If an extent is invalid or allocation fails.
     */
    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine,
                                      SushiRuntime::span<const int64_t> dims,
                                      DType dtype = DType::FLOAT32);

    /**
     * @brief Allocates a tensor of @p shape with its own Storage.
     * @param engine The engine whose context owns the allocator.
     * @param shape The extents.
     * @param dtype The element type.
     * @return The new tensor; nothing else references its Storage.
     * @throws Error As the span overload does.
     */
    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine,
                                      const Shape& shape,
                                      DType dtype = DType::FLOAT32);

    /**
     * @brief Allocates a tensor with @p like's extents, dtype, and layout.
     * @param engine The engine whose context owns the allocator.
     * @param like The tensor to copy the description from.
     * @return The new tensor; it shares nothing with @p like.
     * @throws Error If @p like holds no elements, or the allocation fails.
     */
    SushiBLAS::Tensor allocate_like(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& like);

    /**
     * @brief Allocates one arena buffer for memory planning.
     * @param engine The engine whose context owns the allocator.
     * @param bytes Number of bytes the arena holds; must be positive.
     * @return The newly allocated storage.
     * @throws Error If @p bytes is zero or allocation fails.
     */
    SushiRuntime::sushi_ptr<SushiBLAS::Storage> allocate_arena(SushiBLAS::Engine& engine,
                                                               std::size_t bytes);

    /**
     * @brief Returns a tensor view into @p arena at @p byte_offset.
     * @param engine The engine whose default layout the view takes.
     * @param arena Storage to carve from; must outlive the view.
     * @param byte_offset Offset in bytes; must be aligned to element width.
     * @throws Error If offset is misaligned or out of bounds.
     */
    SushiBLAS::Tensor view_tensor(SushiBLAS::Engine& engine,
                                  const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
                                  std::size_t byte_offset,
                                  const Shape& shape,
                                  DType dtype);

} // namespace SushiAI
