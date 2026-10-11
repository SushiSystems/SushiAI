/**************************************************************************/
/* position_feed.cpp                                                      */
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
 * @file position_feed.cpp
 * @brief Implements the choice of a PositionFeed from the leaves a step is fed.
 * @author Mustafa Garip
 */

#include "position_feed.hpp"

#include <SushiAI/core/check.hpp>
#include <SushiAI/core/status.hpp>
#include "ring_position_feed.hpp"
#include "step_leaf.hpp"
#include "table_position_feed.hpp"

namespace SushiAI::Language::Generation::Detail
{
    Result<std::unique_ptr<PositionFeed>> make_position_feed(
        SushiBLAS::Engine& engine, const Train::EvaluationStep& step,
        const CachedStepLeaves& leaves, std::int64_t window)
    {
        const bool has_table_row = is_fed(step, leaves.position);
        const bool has_ring_slot = is_fed(step, leaves.ring.slot);
        SA_FAIL_IF(has_table_row == has_ring_slot, StatusCode::InvalidDescription,
                   "{}: the step is fed {} of the position leaf '{}' and the cache slot '{}'; "
                   "it reads a step that is fed exactly one of them",
                   K_CACHED_GENERATOR, has_table_row ? "both" : "neither", leaves.position,
                   leaves.ring.slot);

        if (has_ring_slot)
            return RingPositionFeed::make(engine, step, leaves.ring, window);

        return TablePositionFeed::make(engine, step, leaves.position);
    }

} // namespace SushiAI::Language::Generation::Detail
