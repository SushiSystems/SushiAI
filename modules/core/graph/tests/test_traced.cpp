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
 * @file test_traced.cpp
 * @brief Tests Traced and the refusal it carries through a chain of emitters.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

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
    /** @brief Holds what the stand-in builder did. */
    struct TraceLog
    {
        /** @brief Holds how many values were emitted. */
        ValueID emitted = 0;

        /** @brief Holds how many refusals were recorded. */
        int refusals = 0;
    };

    /** @brief Stands in for GraphBuilder: it issues ids and records refusals in a TraceLog. */
    class StandInBuilder final
    {
        public:
            /** @brief Records into @p log, which must outlive the builder. */
            explicit StandInBuilder(TraceLog& log) noexcept : log_(&log)
            {

            }

            /** @brief Issues the next id. */
            [[nodiscard]] ValueID emit() const noexcept
            {
                return log_->emitted++;
            }

            /** @brief Records one refusal and returns @p failure. */
            [[nodiscard]] Status refuse(Status failure) const noexcept
            {
                ++log_->refusals;
                return failure;
            }

        private:
            /** @brief Points at the log this builder writes. */
            TraceLog* log_;
    };

    /** @brief Emits one value over @p operand, refusing an id above @p limit. */
    Traced emit_below(const StandInBuilder& builder, const Traced& operand,
                      ValueID limit) noexcept
    {
        SA_TRY_ASSIGN(const ValueID x, operand);
        SA_REFUSE_IF(builder, x > limit, "operand {} is above the limit {}", x, limit);

        return builder.emit();
    }

    /** @brief Emits one value over @p operand, which may be absent. */
    Traced emit_over(const StandInBuilder& builder, const Traced& operand) noexcept
    {
        SA_TRY_ASSIGN(const ValueID x, operand);
        static_cast<void>(x);

        return builder.emit();
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

TEST(Unit_Traced, DefaultHoldsTheAbsentValueAndNoFailure)
{
    const Traced absent;
    EXPECT_TRUE(absent.has_value());
    EXPECT_TRUE(absent.status().is_ok());
    EXPECT_EQ(absent.value(), SushiAI::Graph::K_NO_VALUE);
}

TEST(Unit_Traced, HoldsAnIDWithASuccessStatus)
{
    const Traced traced(ValueID{12});
    EXPECT_TRUE(traced.has_value());
    SA_EXPECT_OK(traced);
    EXPECT_EQ(traced.value(), ValueID{12});
}

TEST(Unit_Traced, HoldsAFailureAndTheAbsentID)
{
    const Traced traced(description_failure("MATMUL needs two rank-2 operands"));
    EXPECT_FALSE(traced.has_value());
    SA_EXPECT_FAILURE(traced, StatusCode::InvalidDescription, "MATMUL", "rank-2");
}

// A Traced stores its id in place, so an unchanged block count means no allocation.
TEST(Unit_Traced, SuccessPathCreatesNoFailureBlock)
{
    const std::size_t before = live_failure_count();
    TraceLog log;
    const StandInBuilder builder(log);

    const Traced absent;
    const Traced first = emit_over(builder, absent);
    const Traced second = emit_below(builder, first, 8);
    Traced copied(second);
    const Traced moved(std::move(copied));

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, moved);
    EXPECT_EQ(id, ValueID{1});
    EXPECT_EQ(log.refusals, 0);
    EXPECT_EQ(live_failure_count(), before);
}

TEST(Unit_Traced, MovedFromTracedKeepsItsState)
{
    Traced failed(description_failure("window is larger than the image"));
    const Traced taken(std::move(failed));
    SA_EXPECT_FAILURE(taken, StatusCode::InvalidDescription, "window is larger");
    SA_EXPECT_FAILURE(failed, StatusCode::InvalidDescription, "window is larger");

    Traced held(ValueID{4});
    Traced assigned;
    assigned = std::move(held);
    EXPECT_EQ(assigned.value(), ValueID{4});
    EXPECT_EQ(held.value(), ValueID{4});
}

TEST(Unit_Traced, AssignmentReplacesAFailureWithAnID)
{
    Traced traced(description_failure("first attempt"));
    traced = ValueID{3};
    EXPECT_TRUE(traced.has_value());
    EXPECT_EQ(traced.value(), ValueID{3});

    traced = description_failure("second attempt");
    SA_EXPECT_FAILURE(traced, StatusCode::InvalidDescription, "second attempt");
}

TEST(Unit_Traced, AnAbsentOperandPassesThroughAnEmitter)
{
    TraceLog log;
    const StandInBuilder builder(log);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, emit_over(builder, Traced()));
    EXPECT_EQ(id, ValueID{0});
    EXPECT_EQ(log.emitted, ValueID{1});
}

TEST(Unit_Traced, RefusalTravelsThroughAChainOfTwoCallsAndEmitsNothing)
{
    TraceLog log;
    const StandInBuilder builder(log);

    const Traced refused = emit_below(builder, ValueID{9}, 4);
    SA_EXPECT_FAILURE(refused, StatusCode::InvalidDescription,
                      "operand 9 is above the limit 4");
    EXPECT_EQ(log.refusals, 1);
    EXPECT_EQ(log.emitted, ValueID{0});

    const Traced chained = emit_over(builder, emit_below(builder, refused, 100));
    SA_EXPECT_FAILURE(chained, StatusCode::InvalidDescription,
                      "operand 9 is above the limit 4");
    EXPECT_EQ(chained.status().message().data(), refused.status().message().data());
    EXPECT_EQ(log.refusals, 1);
    EXPECT_EQ(log.emitted, ValueID{0});
}

TEST(Unit_Traced, NestedCallsEmitWhenNothingIsRefused)
{
    TraceLog log;
    const StandInBuilder builder(log);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id,
                            emit_over(builder, emit_below(builder, ValueID{2}, 4)));
    EXPECT_EQ(id, ValueID{1});
    EXPECT_EQ(log.emitted, ValueID{2});
    EXPECT_EQ(log.refusals, 0);
}

TEST(Unit_Traced, TryForwardsATracedFailureAsAStatus)
{
    bool reached_end = false;
    SA_EXPECT_FAILURE(forward_traced(description_failure("bias shape"), reached_end),
                      StatusCode::InvalidDescription, "bias shape");
    EXPECT_FALSE(reached_end);

    SA_EXPECT_OK(forward_traced(ValueID{1}, reached_end));
    EXPECT_TRUE(reached_end);
}

TEST(Unit_Traced, ValueOnAFailureIsFatalAndPrintsTheFailure)
{
    const Traced traced(description_failure("stride must be positive"));
    SA_EXPECT_FATAL((void)traced.value(),
                    "Traced::value() was called on a failure", "stride must be positive");
}

TEST(Unit_Traced, BuildingFromASuccessStatusIsFatal)
{
    SA_EXPECT_FATAL((void)Traced(Status()), "a Traced was built from a success Status");
}

/** @brief Tests that GraphBuilder::refuse marks its graph and returns the failure it was given. */
TEST(Unit_Traced, RefuseMarksTheGraphAndReturnsTheFailure)
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

/** @brief Tests that SA_REFUSE_IF over a GraphBuilder marks the graph only when it refuses. */
TEST(Unit_Traced, RefuseIfOverAGraphBuilderMarksTheGraphOnlyOnARefusal)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_ASSERT_OK_AND_ASSIGN(const ValueID id, declare_below(builder, 4, 8));
    EXPECT_EQ(id, ValueID{0});
    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_FALSE(graph.has_refusal());

    SA_EXPECT_FAILURE(declare_below(builder, 9, 8), StatusCode::InvalidDescription,
                      "extent 9 is above the limit 8");
    EXPECT_EQ(graph.value_count(), 1u);
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that observe marks the graph for a failed Status and not for a success. */
TEST(Unit_Traced, ObserveMarksTheGraphForAFailedStatusOnly)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    SA_EXPECT_OK(builder.observe(Status()));
    EXPECT_FALSE(graph.has_refusal());

    SA_EXPECT_FAILURE(builder.observe(description_failure("unknown layer type")),
                      StatusCode::InvalidDescription, "unknown layer type");
    EXPECT_TRUE(graph.has_refusal());
}

/** @brief Tests that observe passes a Result through and marks the graph when it holds a failure. */
TEST(Unit_Traced, ObserveMarksTheGraphForAFailedResultOnly)
{
    SushiAI::Graph::Graph graph;
    const SushiAI::Graph::GraphBuilder builder(graph);

    const SushiAI::Result<int> held = builder.observe(SushiAI::Result<int>(7));
    ASSERT_TRUE(held.has_value());
    EXPECT_EQ(held.value(), 7);
    EXPECT_FALSE(graph.has_refusal());

    const SushiAI::Result<int> failed =
        builder.observe(SushiAI::Result<int>(description_failure("channels do not divide")));
    SA_EXPECT_FAILURE(failed, StatusCode::InvalidDescription, "channels do not divide");
    EXPECT_TRUE(graph.has_refusal());
}
