/**************************************************************************/
/* arena.cpp                                                              */
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

            // The plan sized this slot from the value's own shape and dtype, so
            // a mismatch here means the caller is materialising a value against
            // a description the planner never saw - which would silently spill
            // into the neighbouring tensor's bytes.
            SA_THROW_IF(bytes > placement.bytes,
                        "Value {} was planned {} bytes but a {} {} view needs {}; the shape or "
                        "element type being materialised is not the one that was planned",
                        id, placement.bytes, dtype_name(dtype), shape.to_string(), bytes);

            return view_tensor(*engine_, storages_[placement.arena], placement.offset,
                               shape, dtype);
        }

    } // namespace Graph
} // namespace SushiAI
