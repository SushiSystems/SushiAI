/**************************************************************************/
/* arena.cpp                                                              */
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
 * @file arena.cpp
 * @brief Implements ArenaSet storage allocations and view projections.
 * @author Mustafa Garip
 */

#include <utility>
#include <SushiAI/graph/arena.hpp>

namespace SushiAI
{
    namespace Graph
    {
        ArenaSet::ArenaSet(SushiBLAS::Engine& engine, MemoryPlan plan)
            : engine_(&engine), plan_(std::move(plan))
        {
            storages_.reserve(plan_.arenas().size());
            for (const Arena& arena : plan_.arenas())
                storages_.push_back(allocate_arena(engine, arena.bytes));
        }

        SushiBLAS::Tensor ArenaSet::view(ValueId id, const Shape& shape, DType dtype) const
        {
            const Placement& placement = plan_.placement(id);

            SA_THROW_IF(placement.arena == NO_ARENA,
                        "Value {} was not packed into an arena; it owns its own allocation", id);

            const std::size_t bytes = static_cast<std::size_t>(shape.num_elements())
                                    * dtype_size(dtype);

            // Requested view byte size must not exceed the planned arena slot allocation.
            SA_THROW_IF(bytes > placement.bytes,
                        "Value {} was planned {} bytes but a {} {} view needs {}; the shape or "
                        "element type being materialised is not the one that was planned",
                        id, placement.bytes, dtype_name(dtype), shape.to_string(), bytes);

            return view_tensor(*engine_, storages_[placement.arena], placement.offset,
                               shape, dtype);
        }

    } // namespace Graph
} // namespace SushiAI
