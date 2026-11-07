/**************************************************************************/
/* test_step_recorder.cpp                                                 */
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


/**
 * @file test_step_recorder.cpp
 * @brief Tests that StepRecorderList drives its members as one recorder, in the order added.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

using namespace SushiAI;
using Graph::InMemoryStepRecorder;
using Graph::StepRecorder;
using Graph::StepRecorderList;
using Call = Graph::InMemoryStepRecorder::Call;
using Entry = Graph::InMemoryStepRecorder::Entry;
using Journal = Graph::InMemoryStepRecorder::Journal;

namespace
{
    /** @brief Overrides only the pure call, so both replay hooks are the interface's own. */
    class RecordOnlyRecorder final : public StepRecorder
    {
        public:
            /** @brief Returns a fixed count of three tasks. */
            [[nodiscard]] Result<std::size_t> record_into_step(const Graph::StepRecording&) override
            {
                return 3;
            }
    };

    /** @brief Returns the entries of the members named by @p order, each with @p call. */
    Journal entries_of(const std::vector<int>& order, Call call)
    {
        Journal entries;
        for (const int member : order)
            entries.push_back(Entry{member, call});

        return entries;
    }

    /** @brief Runs each case with the order in which three members are added to a list. */
    class Small_StepRecorderOrder : public ::testing::TestWithParam<std::vector<int>>
    {
        protected:
            /** @brief Adds the members named by the parameter to the list, in that order. */
            void add_members()
            {
                for (const int member : GetParam())
                    SA_EXPECT_OK(list.add(*recorders[member]));
            }

            /// @brief Holds the calls the three members receive, across all of them.
            Journal journal;

            /// @brief Holds the member numbered 0.
            InMemoryStepRecorder a{0, journal, 1};

            /// @brief Holds the member numbered 1.
            InMemoryStepRecorder b{1, journal, 1};

            /// @brief Holds the member numbered 2.
            InMemoryStepRecorder c{2, journal, 1};

            /// @brief Holds the three members by their number.
            StepRecorder* recorders[3] = {&a, &b, &c};

            /// @brief Holds the list under test.
            StepRecorderList list;
    };
} // namespace

/** @brief Tests that a list with no member reports empty and records zero tasks. */
TEST(Small_StepRecorder, EmptyListRecordsNothing)
{
    StepRecorderList list;

    EXPECT_TRUE(list.empty());
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            list.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 0u);
}

/** @brief Tests that adding one recorder makes the list non-empty and calls nothing on it. */
TEST(Small_StepRecorder, AddMakesTheListNonEmpty)
{
    InMemoryStepRecorder a(1);
    StepRecorderList list;

    SA_EXPECT_OK(list.add(a));

    EXPECT_FALSE(list.empty());
    EXPECT_TRUE(a.calls().empty());
}

/** @brief Tests that record_into_step returns the sum of the members' task counts. */
TEST(Small_StepRecorder, RecordSumsTheMemberCounts)
{
    InMemoryStepRecorder a(2);
    InMemoryStepRecorder b(0);
    InMemoryStepRecorder c(5);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            list.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 7u);
}

/** @brief Tests that record_into_step calls each member once, in the order added. */
TEST_P(Small_StepRecorderOrder, RecordCallsMembersInTheOrderAdded)
{
    add_members();

    SA_EXPECT_OK(list.record_into_step(Graph::StepRecording::detached()));

    EXPECT_EQ(journal, entries_of(GetParam(), Call::Record));
}

/** @brief Tests that before_replay calls each member once, in the order added. */
TEST_P(Small_StepRecorderOrder, BeforeReplayCallsMembersInTheOrderAdded)
{
    add_members();

    list.before_replay();

    EXPECT_EQ(journal, entries_of(GetParam(), Call::BeforeReplay));
}

/** @brief Tests that after_replay calls each member once, in the order added. */
TEST_P(Small_StepRecorderOrder, AfterReplayCallsMembersInTheOrderAdded)
{
    add_members();

    list.after_replay();

    EXPECT_EQ(journal, entries_of(GetParam(), Call::AfterReplay));
}

INSTANTIATE_TEST_SUITE_P(Orders, Small_StepRecorderOrder,
                         ::testing::Values(std::vector<int>{0, 1, 2}, std::vector<int>{1, 0},
                                           std::vector<int>{2, 0, 1}));

/** @brief Tests that a member with no replay hooks still records through the list. */
TEST(Small_StepRecorder, MemberWithoutReplayHooksIsDriven)
{
    InMemoryStepRecorder a(4);
    RecordOnlyRecorder plain;
    StepRecorderList list;
    SA_EXPECT_OK(list.add(plain));
    SA_EXPECT_OK(list.add(a));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            list.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 7u);
    list.before_replay();
    list.after_replay();

    EXPECT_EQ(a.calls(), (std::vector<Call>{Call::Record, Call::BeforeReplay, Call::AfterReplay}));
}

/** @brief Tests that a list driven through a StepRecorder reference reaches its members. */
TEST(Small_StepRecorder, ListIsDrivenThroughTheInterface)
{
    InMemoryStepRecorder a(2);
    InMemoryStepRecorder b(3);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    StepRecorder& trailing = list;

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            trailing.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 5u);
    trailing.before_replay();
    trailing.after_replay();

    const std::vector<Call> expected = {Call::Record, Call::BeforeReplay, Call::AfterReplay};
    EXPECT_EQ(a.calls(), expected);
    EXPECT_EQ(b.calls(), expected);
}

/** @brief Tests that a list held as a member of another runs in its place in the outer order. */
TEST(Small_StepRecorder, NestedListKeepsItsPlaceInTheOrder)
{
    Journal journal;
    InMemoryStepRecorder a(0, journal, 1);
    InMemoryStepRecorder b(1, journal, 2);
    InMemoryStepRecorder c(2, journal, 4);
    StepRecorderList inner;
    SA_EXPECT_OK(inner.add(b));
    StepRecorderList outer;
    SA_EXPECT_OK(outer.add(a));
    SA_EXPECT_OK(outer.add(inner));
    SA_EXPECT_OK(outer.add(c));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            outer.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 7u);
    EXPECT_EQ(journal, entries_of({0, 1, 2}, Call::Record));
}

/** @brief Tests that a detached recording refuses to hand out a recorder. */
TEST(Medium_StepRecordingDeathTest, ADetachedRecordingHoldsNoRecorder)
{
    const Graph::StepRecording recording = Graph::StepRecording::detached();

    EXPECT_TRUE(recording.is_detached());
    SA_EXPECT_FATAL((void)recording.tasks(), "detached", "task recorder");
    SA_EXPECT_FATAL((void)recording.graph(), "detached", "runtime recorder");
}

/** @brief Tests that adding a recorder the list already holds is fatal and leaves one entry. */
TEST(Medium_StepRecorderDeathTest, AddRejectsARecorderAlreadyHeld)
{
    InMemoryStepRecorder a(2);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));

    SA_EXPECT_FATAL((void)list.add(a), "StepRecorderList::add", "already");

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count,
                            list.record_into_step(Graph::StepRecording::detached()));
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(a.calls(), (std::vector<Call>{Call::Record}));
}

/** @brief Tests that adding a list to itself is fatal and leaves the list empty. */
TEST(Medium_StepRecorderListDeathTest, AddRejectsTheListItself)
{
    StepRecorderList list;

    SA_EXPECT_FATAL((void)list.add(list), "StepRecorderList::add", "itself");

    EXPECT_TRUE(list.empty());
}
