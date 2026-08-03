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
 * optimizer's momentum buffer, a dataset batch - is allocated through the
 * functions below, and every one of them gets its own @c SushiBLAS::Storage.
 *
 * @section why Why one Storage per tensor, and not one arena
 *
 * This looks like a missed optimisation and is not. Almost every SushiBLAS
 * kernel captures @c tensor.storage->data_ptr and ignores
 * @c tensor.storage_offset - only the reductions, @c logic().all/any,
 * @c random().constant and @c io() honour it. A tensor carved out of a shared
 * arena at a non-zero offset and handed to gemm, axpy or an elementwise op is
 * therefore read from element zero of the arena. The result is a plausible
 * wrong answer, not an error, and it would surface as a model that trains
 * slightly badly.
 *
 * Distinct base pointers also matter to SushiRuntime: its dependency tracker
 * shards its access history on an allocation's base pointer, so a single
 * arena collapses every shard into one chain and turns hazard resolution
 * quadratic. See ir.hpp's @c Lifetime comment.
 *
 * A real arena pass - which is what @c Lifetime exists for - belongs behind a
 * SushiBLAS fix, not in front of one. Until then this is the rule, and the
 * whole library goes through this file so there is exactly one place to
 * change when that fix lands.
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
     * @throws Error If @p dims describes no elements.
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
     * @throws Error If @p shape describes no elements.
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
     * @throws Error If @p like holds no elements.
     */
    SushiBLAS::Tensor allocate_like(SushiBLAS::Engine& engine, const SushiBLAS::Tensor& like);

} // namespace SushiAI
