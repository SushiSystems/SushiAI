/**************************************************************************/
/* ring_position_feed.hpp                                                 */
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
 * @file ring_position_feed.hpp
 * @brief Declares RingPositionFeed, which places a token by its angles and its cache slot.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include "position_feed.hpp"

namespace SushiAI::Language::Generation::Detail
{
    /**
     * @brief Feeds the cache slot of a token, its position modulo T, and the cosines and
     *        sines of its position counted from the last reset.
     *
     * The angles of a row do not change when the window slides, so every cached row stays.
     */
    class RingPositionFeed final : public PositionFeed
    {
        public:
            /**
             * @brief Allocates the feeds of the three leaves of @p leaves.
             * @param window T, the rows of a cache; at least 1.
             * @return The feed, or InvalidDescription for a ring that names no function for
             *         its angles, or a leaf that is absent or does not fit.
             */
            [[nodiscard]] static Result<std::unique_ptr<PositionFeed>> make(
                SushiBLAS::Engine& engine, const Train::EvaluationStep& step,
                const RingStepLeaves& leaves, std::int64_t window);

            RingPositionFeed(const RingPositionFeed&) = delete;
            RingPositionFeed& operator=(const RingPositionFeed&) = delete;
            RingPositionFeed(RingPositionFeed&&) = delete;
            RingPositionFeed& operator=(RingPositionFeed&&) = delete;
            ~RingPositionFeed() override = default;

            /** @brief Writes position modulo T and the angles of @p position. */
            void place(std::int64_t position) noexcept override;

            /** @brief Lists the slot, the cosines and the sines. */
            [[nodiscard]] SushiRuntime::span<const Train::NamedFeed> feeds()
                const noexcept override;

            /** @brief Returns true. */
            [[nodiscard]] bool keeps_rows_on_slide() const noexcept override;

        private:
            /** @brief Counts the leaves place() writes. */
            static constexpr std::size_t K_LEAVES = 3;

            /** @brief Holds the tensors a run feeds. */
            struct Feeds
            {
                Data::Batch slot;   /**< @brief The [1, 1] int32 row of the cache. */
                Data::Batch cosine; /**< @brief The [1, pairs] float32 cosines. */
                Data::Batch sine;   /**< @brief The [1, pairs] float32 sines. */
            };

            /** @brief Takes the tensors make() allocated and the names of their leaves. */
            RingPositionFeed(Feeds feeds, const RingStepLeaves& leaves, std::int64_t window);

            /// @brief Holds the tensors each run feeds.
            Feeds feeds_;

            /// @brief Names the three leaves of the step, in the order of named_.
            std::string names_[K_LEAVES];

            /// @brief Pairs each name with its tensor; it points into this object.
            Train::NamedFeed named_[K_LEAVES];

            /// @brief Fills the cosines and the sines of a position.
            AngleRows angles_;

            /// @brief Holds T, the rows of a cache.
            std::int64_t window_;
    };

} // namespace SushiAI::Language::Generation::Detail
