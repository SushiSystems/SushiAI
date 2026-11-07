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
 * @file test_region_partitioner.cpp
 * @brief Tests how a Session drives the RegionPartitioner it is given.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "eager_run_support.hpp"
#include <SushiAI/failure/status.hpp>
#include <SushiAI/eager/in_memory_region_partitioner.hpp>
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

    /** @brief Expects @p segments to start at node 0 and each to begin where the one before ends. */
    void expect_segments_are_contiguous(const std::vector<Eager::Segment>& segments)
    {
        ASSERT_FALSE(segments.empty());
        EXPECT_EQ(segments.front().begin, Graph::NodeID{0});
        for (std::size_t i = 1; i < segments.size(); ++i)
            EXPECT_EQ(segments[i].begin, segments[i - 1].end) << "segment " << i;
    }

    /** @brief Owns a session over the two-layer model, built on a given partitioner. */
    struct Rig
    {
        /** @brief Builds the executor and the session on @p backend around @p partitioner. */
        Rig(SushiAI::RuntimeExecutionBackend& backend, const Eager::RegionPartitioner& partitioner)
            : executor(backend), session(backend, executor, partitioner), update(K_RATE)
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
    const Eager::ScopePartitioner cutter;
    const Eager::InMemoryRegionPartitioner partitioner(cutter);
    Rig rig(backend(), partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step());
    SA_EXPECT_OK(rig.step());

    EXPECT_EQ(partitioner.calls().size(), 2u);
    EXPECT_EQ(rig.session.step_count(), 2u);
}

/** @brief Tests that the segments handed over cover exactly the forward nodes of the step. */
TEST_F(Medium_RegionPartitioner, TheSegmentsCoverTheForwardNodes)
{
    const Eager::ScopePartitioner cutter;
    const Eager::InMemoryRegionPartitioner partitioner(cutter);
    Rig rig(backend(), partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_OK(rig.step());

    const Eager::InMemoryRegionPartitioner::Call& call = partitioner.calls().front();
    expect_segments_are_contiguous(call.segments);
    EXPECT_EQ(call.segments.back().end, call.forward_node_count);
    EXPECT_LT(call.forward_node_count, call.node_count);
}

/** @brief Tests that a partitioner's thrown refusal reaches the caller and counts no step. */
TEST_F(Medium_RegionPartitioner, AThrownRefusalFailsTheStep)
{
    Eager::InMemoryRegionPartitioner partitioner;
    partitioner.script_refusal("this graph has no cut");
    Rig rig(backend(), partitioner);
    SA_EXPECT_OK(rig.prepare(*engine));

    SA_EXPECT_FAILURE(rig.step(), StatusCode::InvalidDescription, "this graph has no cut");
    EXPECT_EQ(rig.session.step_count(), 0u);
}
