/**************************************************************************/
/* arena.hpp                                                              */
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
 * @file arena.hpp
 * @brief The memory a MemoryPlan asked for, allocated once and carved on demand.
 *
 * The device-lane counterpart of the host-lane planner: @c MemoryPlan decides
 * *where* every value's bytes go, @c ArenaSet is *the bytes*. Keeping the two
 * apart is what lets the planner be unit-tested over synthetic graphs with no
 * engine present, and it puts the whole "allocate once, reuse for ever"
 * property in one small class rather than spread through the pool.
 *
 * Every arena is allocated in the constructor and never again. That is the
 * headline property of the pass - after the first step a training loop
 * allocates nothing - and it is a structural guarantee here rather than a
 * discipline: there is no method on this class that allocates.
 */

#include <cstddef>
#include <vector>
#include <SushiBLAS/engine.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/memory_plan.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @class ArenaSet
         * @brief Owns one Storage per arena in a plan and hands out views into them.
         */
        class ArenaSet
        {
            public:
                /**
                 * @brief Allocates every arena @p plan asks for, once.
                 *
                 * @param engine The engine whose context owns the allocator; must
                 *        outlive this object and every view taken from it.
                 * @param plan The plan to realise; copied, because the views this
                 *        object hands out have to keep matching it for its whole
                 *        life and a caller's plan is free to go out of scope.
                 * @throws Error If an arena allocation fails.
                 */
                ArenaSet(SushiBLAS::Engine& engine, MemoryPlan plan);

                /** @brief The plan these arenas realise. */
                const MemoryPlan& plan() const noexcept { return plan_; }

                /** @brief Whether @p id's bytes live in one of these arenas. */
                bool holds(ValueId id) const noexcept { return plan_.is_packed(id); }

                /**
                 * @brief The view of @p id's slot.
                 *
                 * @param id The value to address.
                 * @param shape Its extents.
                 * @param dtype Its element type.
                 * @return A tensor sharing the arena's Storage at the planned offset.
                 * @throws Error If @p id is not packed, or the view does not fit
                 *         the slot the plan reserved for it - which means the
                 *         shape or dtype being asked for is not the one planned.
                 */
                SushiBLAS::Tensor view(ValueId id, const Shape& shape, DType dtype) const;

                /** @brief How many arenas were allocated. */
                std::size_t arena_count() const noexcept { return storages_.size(); }

                /** @brief Total bytes across every arena. */
                std::size_t bytes() const noexcept { return plan_.arena_bytes(); }

            private:
                SushiBLAS::Engine* engine_;
                MemoryPlan plan_;
                std::vector<SushiRuntime::sushi_ptr<SushiBLAS::Storage>> storages_;
        };

    } // namespace Graph
} // namespace SushiAI
