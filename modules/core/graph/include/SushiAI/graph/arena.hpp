/**************************************************************************/
/* arena.hpp                                                              */
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
 * @file arena.hpp
 * @brief Declares the ArenaSet class for managing planned memory arenas.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
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
                 * @return The arena set, or the OutOfMemory failure of the first arena the
                 *         allocator refuses.
                 */
                [[nodiscard]] static Result<ArenaSet> make(SushiBLAS::Engine& engine,
                                                           MemoryPlan plan);

                /** @brief The plan these arenas realise. */
                const MemoryPlan& plan() const noexcept { return plan_; }

                /** @brief Whether @p id's bytes live in one of these arenas. */
                bool holds(ValueId id) const noexcept { return plan_.is_packed(id); }

                /**
                 * @brief Returns a tensor view of the slot reserved for @p id.
                 * @return A tensor sharing the arena storage at the planned offset.
                 * @pre @p id is packed into an arena, and its bytes fit the planned slot; a
                 *      violation stops the process.
                 */
                SushiBLAS::Tensor view(ValueId id, const Shape& shape, DType dtype) const;

                /** @brief How many arenas were allocated. */
                std::size_t arena_count() const noexcept { return storages_.size(); }

                /** @brief Total bytes across every arena. */
                std::size_t bytes() const noexcept { return plan_.arena_bytes(); }

            private:
                /** @brief One owning handle per arena, in the order of the plan's arenas. */
                using Storages = std::vector<SushiRuntime::sushi_ptr<SushiBLAS::Storage>>;

                /**
                 * @brief Takes the arenas that make() has allocated.
                 * @param engine The engine whose context owns the allocator.
                 * @param plan The memory plan the storages realise.
                 * @param storages One storage per arena of @p plan, in the plan's order.
                 */
                ArenaSet(SushiBLAS::Engine& engine, MemoryPlan plan, Storages storages) noexcept;

                SushiBLAS::Engine* engine_;
                MemoryPlan plan_;
                Storages storages_;
        };

    } // namespace Graph
} // namespace SushiAI
