/**************************************************************************/
/* test_traced.cpp                                                        */
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
 * @file test_traced.cpp
 * @brief Tests Traced and the refusal it carries through a chain of emitters.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>

using SushiAI::Status;
using SushiAI::StatusCode;
using SushiAI::Detail::live_failure_count;
using SushiAI::Graph::Traced;
using SushiAI::Graph::ValueID;

static_assert(std::is_nothrow_default_constructible<Traced>::value, "an absent operand is free");
static_assert(std::is_nothrow_constructible<Traced, ValueID>::value, "holding an id is free");
static_assert(std::is_nothrow_constructible<Traced, Status>::value, "a failure never throws");
static_assert(std::is_convertible<ValueID, Traced>::value, "an id is an operand");
static_assert(std::is_convertible<Status, Traced>::value, "a failure is a result");
static_assert(sizeof(Traced) == 2 * sizeof(void*), "a Traced is an id and one pointer");

namespace
{
    using SushiAI::Graph::Graph;
    using SushiAI::Graph::GraphBuilder;

    /** @brief Holds a graph and the builder that writes it. */
    struct Rig
    {
        Graph graph;          /**< The graph the builder writes. */
        GraphBuilder builder; /**< The builder over @c graph. */

        /** @brief Builds an empty graph and a builder over it. */
        Rig() : builder(graph)
        {

        }
    };

    /** @brief Emits one value over @p operand, refusing an id above @p limit. */
    Traced emit_below(const GraphBuilder& builder, const Traced& operand, ValueID limit) noexcept
    {
        SA_TRY_ASSIGN(const ValueID x, operand);
        SA_REFUSE_IF(builder, x > limit, "operand {} is above the limit {}", x, limit);

        return builder.input(SushiAI::Shape{2}, "v");
    }

    /** @brief Emits one value over @p operand, which may be absent. */
    Traced emit_over(const GraphBuilder& builder, const Traced& operand) noexcept
    {
        SA_TRY_ASSIGN(const ValueID x, operand);
        static_cast<void>(x);

        return builder.input(SushiAI::Shape{2}, "v");
    }

    /** @brief Returns the failure of @p traced, or sets @p reached_end. */
    Status forward_traced(const Traced& traced, bool& reached_end) noexcept
    {
        SA_TRY(traced);
        reached_end = true;
        return Status();
    }

    /** @brief Declares one input over @p builder, refusing an extent above @p limit. */
    Traced declare_below(const SushiAI::Graph::GraphBuilder& builder, int64_t extent,
                         int64_t limit)
    {
        SA_REFUSE_IF(builder, extent > limit, "extent {} is above the limit {}", extent, limit);

        return builder.input(SushiAI::Shape{extent}, "x");
    }

    /** @brief Builds an InvalidDescription failure whose message is @p text. */
    Status description_failure(std::string_view text) noexcept
    {
        return Status::from_message(StatusCode::InvalidDescription, [&]()
        {
            return std::string(text);
        });
    }

} // namespace

TEST(Small_Traced, HoldsAFailureAndTheAbsentID)
{
    const Traced traced(description_failure("MATMUL needs two rank-2 operands"));
    EXPECT_FALSE(traced.has_value());
    SA_EXPECT_FAILURE(traced, StatusCode::InvalidDescription, "MATMUL", "rank-2");
}

/** @brief Tests that a chain of emitters over a success creates no failure block. */
TEST(Small_Traced, SuccessPathCreatesNoFailureBlock)
{
    // A Traced stores its id in place, so an unchanged block count means no allocation.
    const std::size_t before = live_failure_count();
    Rig rig;

    const Traced absent;
    const Traced first = emit_over(rig.builder, absent);
    const Traced second = emit_below(rig.builder, first, 8);
    Traced copied(second);
    const Traced moved(std::move(copied));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, moved);
    EXPECT_EQ(id, ValueID{1});
    EXPECT_FALSE(rig.graph.has_refusal());
    EXPECT_EQ(live_failure_count(), before);
}

/** @brief Tests that a Traced built by move holds the failure the source was built with. */
TEST(Small_Traced, AMovedToTracedHoldsTheFailureItWasBuiltWith)
{
    Traced failed(description_failure("window is larger than the image"));

    const Traced taken(std::move(failed));

    SA_EXPECT_FAILURE(taken, StatusCode::InvalidDescription, "window is larger");
}

/** @brief Tests that move assignment hands the id over to the target. */
TEST(Small_Traced, MoveAssignmentHandsOverTheId)
{
    Traced held(ValueID{4});
    Traced assigned;

    assigned = std::move(held);

    EXPECT_EQ(assigned.value(), ValueID{4});
}

/** @brief Tests that assigning an id to a Traced that holds a failure replaces the failure. */
TEST(Small_Traced, AssigningAnIDReplacesAHeldFailure)
{
    Traced traced(description_failure("first attempt"));

    traced = ValueID{3};

    EXPECT_TRUE(traced.has_value());
    EXPECT_EQ(traced.value(), ValueID{3});
}

/** @brief Tests that assigning a failure to a Traced that holds an id replaces the id. */
TEST(Small_Traced, AssigningAFailureReplacesAHeldID)
{
    Traced traced(ValueID{3});

    traced = description_failure("second attempt");

    SA_EXPECT_FAILURE(traced, StatusCode::InvalidDescription, "second attempt");
}

/** @brief Tests that an absent operand passes through an emitter, which emits one value. */
TEST(Small_Traced, AnAbsentOperandPassesThroughAnEmitter)
{
    Rig rig;

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, emit_over(rig.builder, Traced()));

    EXPECT_EQ(id, ValueID{0});
    EXPECT_EQ(rig.graph.value_count(), 1u);
}

/** @brief Tests that a refusal travels through a chain of emitters and emits nothing. */
TEST(Small_Traced, RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing)
{
    Rig rig;

    const Traced refused = emit_below(rig.builder, ValueID{9}, 4);
    SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription,
                      "operand 9 is above the limit 4");
    EXPECT_TRUE(rig.graph.has_refusal());

    const Traced chained = emit_over(rig.builder, emit_below(rig.builder, refused, 100));
    SA_EXPECT_FAILURE(chained, StatusCode::InvalidDescription,
                      "operand 9 is above the limit 4");
    EXPECT_EQ(rig.graph.value_count(), 0u);
}

/** @brief Tests that nested emitter calls emit when nothing is refused. */
TEST(Small_Traced, NestedCallsEmitWhenNothingIsRefused)
{
    Rig rig;

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id,
                            emit_over(rig.builder, emit_below(rig.builder, ValueID{2}, 4)));

    EXPECT_EQ(id, ValueID{1});
    EXPECT_EQ(rig.graph.value_count(), 2u);
    EXPECT_FALSE(rig.graph.has_refusal());
}

/** @brief Tests that SA_TRY forwards a traced failure as a status. */
TEST(Small_Traced, TryForwardsATracedFailureAsAStatus)
{
    bool reached_end = false;

    SA_EXPECT_FAILURE(forward_traced(description_failure("bias shape"), reached_end),
                      StatusCode::InvalidDescription, "bias shape");
    EXPECT_FALSE(reached_end);
}

/** @brief Tests that SA_TRY passes a successful traced value on. */
TEST(Small_Traced, TryPassesASuccessfulTracedValueOn)
{
    bool reached_end = false;

    SA_EXPECT_OK(forward_traced(ValueID{1}, reached_end));
    EXPECT_TRUE(reached_end);
}

TEST(Small_Traced, ValueOnAFailureIsFatalAndPrintsTheFailure)
{
    const Traced traced(description_failure("stride must be positive"));
    SA_EXPECT_FATAL((void)traced.value(),
                    "Traced::value() was called on a failure", "stride must be positive");
}

TEST(Small_Traced, BuildingFromASuccessStatusIsFatal)
{
    SA_EXPECT_FATAL((void)Traced(Status()), "a Traced was built from a success Status");
}

/** @brief Tests that GraphBuilder::refuse marks its graph and returns the failure it was given. */
TEST(Small_Traced, RefuseMarksTheGraphAndReturnsTheFailure)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);
    ASSERT_FALSE(graph.has_refusal());

    const Status failure = description_failure("kernel is wider than the image");
    const Status returned = builder.refuse(failure);

    SA_EXPECT_FAILURE(returned, StatusCode::InvalidDescription, "kernel is wider");
    EXPECT_EQ(returned.message().data(), failure.message().data());
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that SA_REFUSE_IF over a builder does not mark the graph on success. */
TEST(Small_Traced, RefuseIfOverAGraphBuilderDoesNotMarkTheGraphOnSuccess)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, declare_below(builder, 4, 8));

    EXPECT_EQ(id, ValueID{0});
    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that SA_REFUSE_IF over a builder marks the graph and emits nothing on refusal. */
TEST(Small_Traced, RefuseIfOverAGraphBuilderMarksTheGraphAndEmitsNothingOnRefusal)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_EXPECT_FAILURE(declare_below(builder, 9, 8), StatusCode::InvalidDescription,
                      "extent 9 is above the limit 8");

    EXPECT_EQ(graph.value_count(), 0u);
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that observe does not mark the graph for a success Status. */
TEST(Small_Traced, ObserveDoesNotMarkTheGraphForASuccessStatus)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_EXPECT_OK(builder.observe(Status()));

    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that observe marks the graph for a failed Status. */
TEST(Small_Traced, ObserveMarksTheGraphForAFailedStatus)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_EXPECT_FAILURE(builder.observe(description_failure("unknown layer type")),
                      StatusCode::InvalidDescription, "unknown layer type");

    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that observe passes a successful Result through without marking the graph. */
TEST(Small_Traced, ObservePassesASuccessfulResultThroughWithoutMarkingTheGraph)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    const SushiAI::Result<int> held = builder.observe(SushiAI::Result<int>(7));

    ASSERT_TRUE(held.has_value());
    EXPECT_EQ(held.value(), 7);
    EXPECT_FALSE(graph.has_refusal());
}

/** @brief Tests that observe marks the graph for a failed Result. */
TEST(Small_Traced, ObserveMarksTheGraphForAFailedResult)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    const SushiAI::Result<int> failed =
        builder.observe(SushiAI::Result<int>(description_failure("channels do not divide")));

    SA_EXPECT_FAILURE(failed, StatusCode::InvalidDescription, "channels do not divide");
    EXPECT_TRUE(graph.has_refusal());
}
