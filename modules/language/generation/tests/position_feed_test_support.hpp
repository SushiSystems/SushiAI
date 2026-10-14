/**************************************************************************/
/* position_feed_test_support.hpp                                         */
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
 * @file position_feed_test_support.hpp
 * @brief Defines the one-row carrying step the position feed tests bind their feeds to.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/SushiAI.h>
#include <SushiAI/engine_fixture/engine_test.hpp>

/** @brief Holds what the tests of the position feeds share. */
namespace SushiAI::Language::Generation::Support
{
    /** @brief Holds the columns of the model input and of each angle leaf. */
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

    /** @brief Declares the fed leaves its two flags name and carries a sum that reads them. */
    struct FedRow
    {
        /** @brief Says whether the trace declares K_FEED_POSITION. */
        bool has_position = false;

        /** @brief Says whether the trace declares the slot, the cosines and the sines. */
        bool has_ring = false;

        /** @brief Lists no state. */
        static constexpr auto fields()
        {
            return std::tuple<>{};
        }

        /** @brief Adds the row its index leaf @p name writes of @p value to @p sum. */
        [[nodiscard]] static Result<Graph::ValueID> add_indexed(const NN::Trace& trace,
                                                                std::string_view name,
                                                                Graph::ValueID value,
                                                                Graph::ValueID sum)
        {
            const Graph::GraphBuilder& builder = trace.builder();
            SA_TRY_ASSIGN(const Graph::ValueID index,
                          trace.feed(name, Shape{1, 1}, DType::Int32));
            SA_TRY_ASSIGN(const Graph::ValueID rows, NN::Detail::bridge_throwing([&]()
            {
                return builder.reshape(value, Shape{1, 1, K_FEED_PAIRS});
            }));

            const Graph::Traced placed = builder.scatter_add_rows(rows, index, 1);
            SA_TRY(placed.status());

            return NN::Detail::bridge_throwing([&]()
            {
                return builder.add(sum, placed.value());
            });
        }

        /** @brief Traces the carried sum over every declared leaf, or returns the failure. */
        [[nodiscard]] Result<Graph::ValueID> declare(const NN::Trace& trace,
                                                     Graph::ValueID x) const
        {
            const Graph::GraphBuilder& builder = trace.builder();
            const Shape row{1, K_FEED_PAIRS};
            SA_TRY_ASSIGN(const Graph::ValueID held,
                          trace.carried_leaf(row, "held", DType::Float32));
            SA_TRY_ASSIGN(Graph::ValueID sum, NN::Detail::bridge_throwing([&]()
            {
                return builder.add(held, x);
            }));

            if (has_position)
            {
                SA_TRY_ASSIGN(sum, add_indexed(trace, K_FEED_POSITION, x, sum));
            }

            if (has_ring)
            {
                SA_TRY_ASSIGN(const Graph::ValueID cosine,
                              trace.feed(K_FEED_COSINE, row, DType::Float32));
                SA_TRY_ASSIGN(const Graph::ValueID sine,
                              trace.feed(K_FEED_SINE, row, DType::Float32));
                SA_TRY_ASSIGN(const Graph::ValueID turned, NN::Detail::bridge_throwing([&]()
                {
                    return builder.add(builder.mul(x, cosine), sine);
                }));
                SA_TRY_ASSIGN(sum, add_indexed(trace, K_FEED_SLOT, turned, sum));
            }

            SA_TRY(trace.carry(held, sum));

            return sum;
        }

        /** @brief Traces declare(), or fails the trace and returns @p x. */
        Graph::ValueID forward(const NN::Trace& trace, Graph::ValueID x)
        {
            return trace.value_or_fail(declare(trace, x), x);
        }

        /** @brief Returns @p input. */
        Shape output_shape(const Shape& input) const
        {
            return input;
        }
    };

    static_assert(NN::TraceModule<FedRow>, "FedRow must be a TraceModule");

    /** @brief Compiles the carrying step of one FedRow on the Engine of the test. */
    class FedStepTest : public ::SushiAI::Test::EngineTest
    {
        protected:
            /** @brief Releases the step before the engine it was built on. */
            void TearDown() override
            {
                step.reset();
                pool.reset();
                state.reset();
                ::SushiAI::Test::EngineTest::TearDown();
            }

            /** @brief Builds the step of a FedRow with the two flags; fails the test if not. */
            void build_step(bool has_position, bool has_ring)
            {
                model.has_position = has_position;
                model.has_ring = has_ring;
                SA_ASSERT_OK_AND_ASSIGN(
                    const Train::TracedStep traced,
                    Train::build_carrying_graph(model, Shape{1, K_FEED_PAIRS}));
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(
                    std::unique_ptr<Train::EvaluationStep> compiled,
                    Train::EvaluationStep::make(*engine, traced, *state, *pool));
                step = std::move(compiled);
            }

            /// @brief Declares the leaves of the step.
            FedRow model;

            /// @brief Holds the state of the model, which has no entry.
            std::unique_ptr<NN::ModelState> state;

            /// @brief Holds every value of the step.
            std::unique_ptr<Graph::TensorPool> pool;

            /// @brief Holds the compiled step the feeds are bound to; null until build_step().
            std::unique_ptr<Train::EvaluationStep> step;
    };

} // namespace SushiAI::Language::Generation::Support
