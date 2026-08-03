/**************************************************************************/
/* allocation.hpp                                                         */
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

#pragma once

/**
 * @file allocation.hpp
 * @brief How SushiAI obtains a tensor, stated once.
 *
 * Every tensor in this library - a graph value, a helper constant, an
 * optimizer's momentum buffer, a dataset batch - comes from this file, by one
 * of two routes: its own @c SushiBLAS::Storage (@c allocate_tensor,
 * @c allocate_like), or a view into an arena the memory planner sized
 * (@c allocate_arena, @c view_tensor).
 *
 * @section why Why the second route exists now and did not before
 *
 * This header used to say "one Storage per tensor, always", and the reason it
 * gave was real: almost every SushiBLAS kernel captured
 * @c tensor.storage->data_ptr and ignored @c tensor.storage_offset, so a
 * tensor carved out of a shared arena at a non-zero offset was read from
 * element zero of the arena - a plausible wrong answer rather than an error.
 *
 * **That reason is gone.** Every SushiBLAS op family honours
 * @c storage_offset, @c Storage::data_ptr is private so the offset-dropping
 * expression can no longer be written, and @c Tensor::region() reports the
 * view's own byte interval instead of the bare allocation base - which is what
 * lets two disjoint tensors in one arena be seen as not conflicting by
 * SushiRuntime's dependency tracker instead of falsely serialising.
 *
 * What has *not* changed is the second half of the old rationale. Distinct
 * base pointers still matter to that tracker, which shards its access history
 * on them, so the planner uses one arena per lifetime class rather than one
 * arena for everything, and leaves the persistent classes unpacked entirely.
 * That decision and its reasons live in graph/memory_plan.hpp.
 *
 * @section offsets Offsets are counted in elements
 *
 * @c Tensor::storage_offset is an element count, not a byte count, which is
 * what makes a single-dtype view naturally aligned by construction. A caller
 * mixing dtypes in one arena has to round each start up to its own element
 * width itself; @c view_tensor takes a byte offset and refuses one that is not
 * a whole number of elements, so that mistake is a thrown error rather than a
 * silently shifted tensor.
 *
 * Device lane: this @c core/ header reaches SushiBLAS, unlike its neighbours.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>

namespace SushiAI
{
    /**
     * @brief Allocates a tensor of @p dims with its own Storage.
     *
     * Shared USM, zero offset, the engine's default layout.
     *
     * @param engine The engine whose context owns the allocator.
     * @param dims The extents, outermost first.
     * @param dtype The element type.
     * @return The new tensor; nothing else references its Storage.
     * @throws Error If an extent is not positive, the extents multiply to more
     *         than an int64_t (or their bytes to more than a size_t) can hold,
     *         or the underlying allocation fails - in which case the message
     *         names the shape, the element type and the byte count, because
     *         "allocation failed for N bytes" alone does not identify which of
     *         a model's tensors asked for them.
     */
    SushiBLAS::Tensor allocate_tensor(SushiBLAS::Engine& engine,
                                      std::span<const int64_t> dims,
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
     * @brief Allocates a tensor with @p like's extents, dtype and layout.
     *
     * How an optimizer sizes its state: matching the parameter exactly is not
     * a convenience but a requirement, because SushiBLAS rejects a mixed-dtype
     * axpy and derives a matrix's leading dimension from its shape.
     *
     * @param engine The engine whose context owns the allocator.
     * @param like The tensor to copy the description from.
     * @return The new tensor; it shares nothing with @p like.
     * @throws Error If @p like holds no elements, or the allocation fails.
     */
    SushiBLAS::Tensor allocate_like(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& like);

    /**
     * @brief Allocates one arena: a bare run of bytes for the planner to carve up.
     *
     * Shared USM, like every other allocation here, and aligned by the
     * allocator to @c SushiRuntime::Core::DEFAULT_ALIGNMENT - which is what
     * lets @c view_tensor hand out views that are as well aligned as a
     * standalone tensor would have been.
     *
     * @param engine The engine whose context owns the allocator.
     * @param bytes How many bytes the arena holds; must be positive.
     * @return The storage; nothing references it yet.
     * @throws Error If @p bytes is zero or the underlying allocation fails.
     */
    SushiRuntime::sushi_ptr<SushiBLAS::Storage> allocate_arena(SushiBLAS::Engine& engine,
                                                               std::size_t bytes);

    /**
     * @brief A tensor of @p shape living at @p byte_offset inside @p arena.
     *
     * The one place a non-zero @c storage_offset is constructed. It takes a
     * byte offset because that is what a planner packing mixed dtypes computes,
     * and converts it to the element count @c Tensor stores - refusing an
     * offset that is not a whole number of elements, because such an offset
     * cannot be represented and would otherwise be truncated into a tensor
     * that quietly starts in the wrong place.
     *
     * @param engine The engine whose default layout the view takes.
     * @param arena The storage to carve from; must outlive every view of it.
     * @param byte_offset Where the view's first element starts, in bytes.
     * @param shape The view's extents.
     * @param dtype The view's element type.
     * @return The view; it shares @p arena's Storage.
     * @throws Error If @p arena is null, @p byte_offset is not a multiple of
     *         the element width, or the view would run past the arena.
     */
    SushiBLAS::Tensor view_tensor(SushiBLAS::Engine& engine,
                                  const SushiRuntime::sushi_ptr<SushiBLAS::Storage>& arena,
                                  std::size_t byte_offset,
                                  const Shape& shape,
                                  DType dtype);

} // namespace SushiAI
