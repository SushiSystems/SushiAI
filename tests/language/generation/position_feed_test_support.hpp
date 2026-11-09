/**************************************************************************/
/* position_feed_test_support.hpp                                        */
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
 * @file position_feed_test_support.hpp
 * @brief Defines the fed leaves, the host engine and the in-memory table step that the tests of the
 *        position feeds and of the cached generator bind to.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/in_memory_fed_leaves.hpp>
#include <SushiAI/language/generation/in_memory_one_position_step.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/steps.hpp>

/** @brief Holds what the tests of the position feeds share. */
namespace SushiAI::Language::Generation::Support
{
    /** @brief Holds the columns of each angle leaf. */
    inline constexpr std::int64_t K_FEED_PAIRS = 3;

    /** @brief Holds T, the rows of the cache the ring feeds of these tests wrap over. */
    inline constexpr std::int64_t K_FEED_WINDOW = 4;

    /** @brief Names the [1, 1] int32 leaf of a table position. */
    inline constexpr std::string_view K_FEED_POSITION = "feed_position";

    /** @brief Names the [1, 1] int32 leaf of a ring slot. */
    inline constexpr std::string_view K_FEED_SLOT = "feed_slot";

    /** @brief Names the [1, K_FEED_PAIRS] float32 leaf of the cosines. */
    inline constexpr std::string_view K_FEED_COSINE = "feed_cosine";

    /** @brief Names the [1, K_FEED_PAIRS] float32 leaf of the sines. */
    inline constexpr std::string_view K_FEED_SINE = "feed_sine";

    /** @brief Writes position + pair as each cosine and its negation as each sine. */
    inline void write_counted_row(std::int64_t position, SushiRuntime::span<float> cosines,
                                  SushiRuntime::span<float> sines) noexcept
    {
        for (std::size_t pair = 0; pair < cosines.size(); ++pair)
        {
            cosines[pair] = static_cast<float>(position) + static_cast<float>(pair);
            sines[pair] = -cosines[pair];
        }
    }

    /** @brief Returns the leaves of this file's names, with write_counted_row for the angles. */
    inline CachedStepLeaves feed_leaves()
    {
        return CachedStepLeaves{
            K_FEED_POSITION, "feed_mask", 0.0,
            RingStepLeaves{K_FEED_SLOT, K_FEED_COSINE, K_FEED_SINE, &write_counted_row}};
    }

    /** @brief Returns the leaf @p name of [1, @p columns] and the type @p dtype. */
    inline NamedLeafFacts leaf_of(std::string_view name, std::int64_t columns, DType dtype)
    {
        return NamedLeafFacts{std::string(name), LeafFacts{Shape{1, columns}, dtype}};
    }

    /**
     * @brief Returns the leaves a step is fed when it places a token by a table position
     *        (@p has_position) or by a ring slot with its angles (@p has_ring), or by both or
     *        neither.
     */
    inline InMemoryFedLeaves fed_leaves_of(bool has_position, bool has_ring)
    {
        std::vector<NamedLeafFacts> leaves;
        if (has_position)
            leaves.push_back(leaf_of(K_FEED_POSITION, 1, DType::Int32));

        if (has_ring)
        {
            leaves.push_back(leaf_of(K_FEED_SLOT, 1, DType::Int32));
            leaves.push_back(leaf_of(K_FEED_COSINE, K_FEED_PAIRS, DType::Float32));
            leaves.push_back(leaf_of(K_FEED_SINE, K_FEED_PAIRS, DType::Float32));
        }

        return InMemoryFedLeaves(std::move(leaves));
    }

    /** @brief Holds T, the rows of the cache of the in-memory table step. */
    inline constexpr std::int64_t K_STEP_WINDOW = 8;

    /** @brief Holds V, the scores of a position of the in-memory table step. */
    inline constexpr std::int64_t K_STEP_VOCABULARY = 32;

    /** @brief Names the output the in-memory table step publishes its logits under. */
    inline constexpr std::string_view K_STEP_OUTPUT = "logits";

    /** @brief Names the [1, T] float32 mask leaf of the in-memory table step. */
    inline constexpr std::string_view K_STEP_MASK = "feed_mask";

    /** @brief Returns the leaves a cached generator is given for the in-memory table step. */
    inline CachedStepLeaves table_step_leaves()
    {
        return CachedStepLeaves{K_FEED_POSITION, K_STEP_MASK, 0.0};
    }

    /**
     * @brief Returns an in-memory step of @p window rows that is fed the id, the mask and a table
     *        position, and scores the successor of each id.
     */
    inline std::unique_ptr<InMemoryOnePositionStep> table_step(std::int64_t window,
                                                               std::int64_t vocabulary)
    {
        std::vector<NamedLeafFacts> leaves;
        leaves.push_back(leaf_of(Train::K_MODEL_INPUT_NAME, 1, DType::Int32));
        leaves.push_back(leaf_of(K_STEP_MASK, window, DType::Float32));
        leaves.push_back(leaf_of(K_FEED_POSITION, 1, DType::Int32));

        return std::make_unique<InMemoryOnePositionStep>(std::move(leaves),
                                                         std::string(K_STEP_OUTPUT), vocabulary,
                                                         true);
    }

    /** @brief Gives each test a host engine to allocate the tensors of its feeds on. */
    class HostFeedTest : public ::testing::Test
    {
        protected:
            /** @brief Returns the host engine of the test. */
            [[nodiscard]] SushiBLAS::Engine& engine() noexcept
            {
                return backend_.engine();
            }

        private:
            /// @brief Owns the host engine.
            HostExecutionBackend backend_;
    };

} // namespace SushiAI::Language::Generation::Support
