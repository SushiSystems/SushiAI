/**************************************************************************/
/* test_region_partitioner.cpp                                            */
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
 * @file test_region_partitioner.cpp
 * @brief Tests how a Session drives the RegionPartitioner it is given.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "eager_run_support.hpp"
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_executor.hpp>
#include <SushiAI/eager/region_partitioner.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/segment.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>

using namespace SushiAI;
using Graph::ValueID;

// SaTest stays reachable inside a TEST_F body; see tests/core/eager/README.md.
namespace SaTest = SushiAI::Test;
namespace Support = SushiAI::Eager::Support;

namespace
{
    /** @brief Provides the Engine every case runs on. */
    class Medium_RegionPartitioner : public SaTest::EngineTest
    {
    };

    /** @brief Holds the step size of the update behind every step. */
    constexpr float K_RATE = 0.05f;

    /** @brief Holds the widths of the two-layer model: 6 inputs, 4 hidden, 3 results. */
    const std::vector<int64_t> K_TWO_LAYERS{6, 4, 3};

    /** @brief Cuts a step as ScopePartitioner does and records what each call was given. */
    class RecordingPartitioner final : public Eager::RegionPartitioner
    {
        public:
            /** @brief Records the call, then partitions through ScopePartitioner. */
            [[nodiscard]] Eager::Partition assign(
                const Graph::Graph& graph, SushiRuntime::span<const Eager::Segment> segments,
                const Autograd::Provenance& provenance) const override
            {
                ++calls;
                segments_seen.assign(segments.begin(), segments.end());
                node_count = graph.node_count();
                forward_node_count = provenance.forward_node_count;
                return Eager::ScopePartitioner{}.assign(graph, segments, provenance);
            }

            mutable std::size_t calls = 0;                   /**< How often assign() ran. */
            mutable std::vector<Eager::Segment> segments_seen; /**< The last call's segments. */
            mutable std::size_t node_count = 0;              /**< The last graph's node count. */
            mutable std::size_t forward_node_count = 0;      /**< The last forward node count. */
    };

    /** @brief Refuses every step by throwing, as a partitioner does on a graph it cannot cut. */
    class RefusingPartitioner final : public Eager::RegionPartitioner
    {
        public:
            /** @brief Throws std::invalid_argument naming the refusal. */
            [[nodiscard]] Eager::Partition assign(const Graph::Graph&,
                                                  SushiRuntime::span<const Eager::Segment>,
                                                  const Autograd::Provenance&) const override
            {
                throw std::invalid_argument("this graph has no cut");
            }
    };

    /** @brief Owns a session over the two-layer model, built on a given partitioner. */
    struct Rig
    {
        /** @brief Builds the executor and the session on @p engine around @p partitioner. */
        Rig(SushiBLAS::Engine& engine, const Eager::RegionPartitioner& partitioner)
            : executor(engine), session(engine, executor, partitioner), update(K_RATE)
        {

        }

        /** @brief Allocates the model and one batch and binds every state to the session. */
        [[nodiscard]] Status prepare(SushiBLAS::Engine& engine)
        {
            SA_TRY_ASSIGN(model, Support::make_model(engine, K_TWO_LAYERS));
            SA_TRY_ASSIGN(inputs, Support::make_batch(engine, K_TWO_LAYERS.front(), 0.25));
            SA_TRY_ASSIGN(targets, Support::make_batch(engine, K_TWO_LAYERS.back(), 1.75));
            return Support::bind_model(session, model, update);
        }

        /** @brief Traces one scoped, differentiated step and runs it with the update. */
        [[nodiscard]] Status step()
        {
            Support::TraceOptions options;
            options.scoping = Support::Scoping::PerLayer;
            Eager::Step step = session.begin_step();
            const Graph::Traced loss =
                Support::trace_model(step, model, inputs, targets, options);
            SA_TRY(step.backward(loss));
            return step.apply(update);
        }

        Eager::RegionExecutor executor;          /**< Runs each step. */
        Eager::Session session;                  /**< The session under test. */
        Support::PlainDescent update;            /**< The update behind each step. */
        std::vector<Support::LinearState> model; /**< The layers, bound to the session. */
        SushiBLAS::Tensor inputs;                /**< The [K_ROWS, 6] batch. */
        SushiBLAS::Tensor targets;               /**< The [K_ROWS, 3] targets. */
    };
} // namespace

/** @brief Tests that the session asks its partitioner once per step it runs. */
TEST_F(Medium_RegionPartitioner, TheSessionAsksOncePerStep)
{
    RecordingPartitioner partitioner;
    Rig rig(*engine, partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step());
    SA_EXPECT_OK(rig.step());

    EXPECT_EQ(partitioner.calls, 2u);
    EXPECT_EQ(rig.session.step_count(), 2u);
}

/** @brief Tests that the segments handed over cover exactly the forward nodes of the step. */
TEST_F(Medium_RegionPartitioner, TheSegmentsCoverTheForwardNodes)
{
    RecordingPartitioner partitioner;
    Rig rig(*engine, partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step());

    const std::vector<Eager::Segment>& segments = partitioner.segments_seen;
    ASSERT_FALSE(segments.empty());
    EXPECT_EQ(segments.front().begin, Graph::NodeID{0});
    EXPECT_EQ(segments.back().end, partitioner.forward_node_count);
    for (std::size_t i = 1; i < segments.size(); ++i)
        EXPECT_EQ(segments[i].begin, segments[i - 1].end) << "segment " << i;

    EXPECT_LT(partitioner.forward_node_count, partitioner.node_count);
}

/** @brief Tests that a partitioner's thrown refusal reaches the caller and counts no step. */
TEST_F(Medium_RegionPartitioner, AThrownRefusalFailsTheStep)
{
    const RefusingPartitioner partitioner;
    Rig rig(*engine, partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_FAILURE(rig.step(), StatusCode::InvalidDescription, "this graph has no cut");
    EXPECT_EQ(rig.session.step_count(), 0u);
}
