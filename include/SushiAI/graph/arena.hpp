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

/**
 * @file arena.hpp
 * @brief Declares the ArenaSet class for managing planned memory arenas.
 * @author Mustafa Garip
 */

#pragma once

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
                 * @brief Allocates every arena requested by @p plan.
                 * @param engine The engine whose context owns the allocator.
                 * @param plan The memory plan to realise.
                 * @throws Error If an arena allocation fails.
                 */
                ArenaSet(SushiBLAS::Engine& engine, MemoryPlan plan);

                /** @brief The plan these arenas realise. */
                const MemoryPlan& plan() const noexcept { return plan_; }

                /** @brief Whether @p id's bytes live in one of these arenas. */
                bool holds(ValueId id) const noexcept { return plan_.is_packed(id); }

                /**
                 * @brief Returns a tensor view of the slot reserved for @p id.
                 * @param id The value to address.
                 * @param shape The value extents.
                 * @param dtype The element data type.
                 * @return A tensor sharing the arena storage at the planned offset.
                 * @throws Error If @p id is not packed or size mismatches the slot.
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
