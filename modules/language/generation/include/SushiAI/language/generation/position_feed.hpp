/**************************************************************************/
/* position_feed.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file position_feed.hpp
 * @brief Declares PositionFeed, the seam through which a cached generation places a token.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/fed_leaves.hpp>
#include <SushiAI/train/evaluation_step.hpp>

namespace SushiAI::Language::Generation::Detail
{
    /** @brief Fills the leaves that tell a one-position step where its token stands. */
    class PositionFeed
    {
        public:
            /** @brief Releases the feed through its base. */
            virtual ~PositionFeed() = default;

            /**
             * @brief Writes the leaves of the token at @p position.
             * @param position Counted from the last reset of the step; 0 or more.
             */
            virtual void place(std::int64_t position) noexcept = 0;

            /** @brief Lists the leaves place() writes, each under the name the step reads. */
            [[nodiscard]] virtual SushiRuntime::span<const Train::NamedFeed> feeds()
                const noexcept = 0;

            /** @brief Reports whether a cached row stays right after the window slides. */
            [[nodiscard]] virtual bool keeps_rows_on_slide() const noexcept = 0;
    };

    /**
     * @brief Builds the feed of a step: a table position when @p fed lists the position
     *        leaf of @p leaves, a ring slot when it lists the slot leaf of its ring.
     * @param fed The leaves the step is fed.
     * @param window T, the rows of a cache.
     * @return The feed, or InvalidDescription for a step fed both leaves or neither, or
     *         for a leaf that does not fit.
     */
    [[nodiscard]] Result<std::unique_ptr<PositionFeed>> make_position_feed(
        SushiBLAS::Engine& engine, const FedLeaves& fed, const CachedStepLeaves& leaves,
        std::int64_t window);

} // namespace SushiAI::Language::Generation::Detail
