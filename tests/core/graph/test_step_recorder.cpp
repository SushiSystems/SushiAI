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
/**************************************************************************/


/**
 * @file test_step_recorder.cpp
 * @brief Tests that StepRecorderList drives its members as one recorder, in the order added.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/in_memory_step_recorder.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>

using namespace SushiAI;
using Graph::InMemoryStepRecorder;
using Graph::StepRecorder;
using Graph::StepRecorderList;
using Call = Graph::InMemoryStepRecorder::Call;

namespace
{
    /**
     * @brief Writes each call it receives to a shared journal as "<name>.<call>"; the order across
     * members is what the in-memory adapter, which journals per member, cannot show.
     */
    class JournalRecorder final : public StepRecorder
    {
        public:
            /** @brief Binds the recorder to @p journal, which must outlive it. */
            JournalRecorder(std::string name, std::size_t tasks, std::vector<std::string>& journal)
                : name_(std::move(name)), tasks_(tasks), journal_(&journal)
            {

            }

            /** @brief Journals the call and returns the task count given at construction. */
            [[nodiscard]] Result<std::size_t> record_into_step(const Graph::StepRecording&) override
            {
                journal_->push_back(name_ + ".record");
                return tasks_;
            }

            /** @brief Journals the call. */
            void before_replay() override
            {
                journal_->push_back(name_ + ".before");
            }

            /** @brief Journals the call. */
            void after_replay() override
            {
                journal_->push_back(name_ + ".after");
            }

        private:
            /// @brief Prefix of every journal entry this recorder writes.
            std::string name_;

            /// @brief Count record_into_step() reports.
            std::size_t tasks_;

            /// @brief Journal shared with the other recorders of the test; not owned.
            std::vector<std::string>* journal_;
    };

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

    /** @brief Holds the journal entries three recorders named a, b and c write for @p call. */
    std::vector<std::string> in_order_added(const std::string& call)
    {
        return {"a." + call, "b." + call, "c." + call};
    }

    /**
     * @brief Adds the recorders a, b and c named by @p added to a list in that order, records the
     * list into @p recording and expects the journal to be @p expected.
     */
    void expect_record_order(const Graph::StepRecording& recording,
                             const std::vector<std::size_t>& added,
                             const std::vector<std::string>& expected)
    {
        std::vector<std::string> journal;
        JournalRecorder a("a", 1, journal);
        JournalRecorder b("b", 1, journal);
        JournalRecorder c("c", 1, journal);
        StepRecorder* const recorders[] = {&a, &b, &c};
        StepRecorderList list;
        for (const std::size_t index : added)
            SA_EXPECT_OK(list.add(*recorders[index]));

        SA_EXPECT_OK(list.record_into_step(recording));

        EXPECT_EQ(journal, expected);
    }

    /** @brief Gives each test a graph and a recorder over it, on the shared runtime. */
    class Medium_StepRecorder : public ::testing::Test
    {
        protected:
            /** @brief Builds the graph, its scratch pool and the recorder the tests use. */
            Medium_StepRecorder()
                : late_counts_(SushiAI::Test::TestEnvironment::runtime()),
                  graph_(SushiAI::Test::TestEnvironment::runtime().graph()),
                  scratch_(SushiAI::Test::TestEnvironment::runtime()),
                  recorder_(graph_.recorder(), scratch_, late_counts_),
                  recording_(recorder_, graph_.recorder())
            {

            }

            /// @brief Keeps the late-count tasks of graph_; declared first so it dies last.
            SushiBLAS::LateCountHold late_counts_;

            /// @brief Graph the recorder writes to; no test runs it.
            SushiRuntime::Graph graph_;

            /// @brief Scratch pool the recorder leases from.
            SushiBLAS::ScratchPool scratch_;

            /// @brief Task recorder over graph_.
            SushiBLAS::TaskRecorder recorder_;

            /// @brief Recording every test passes to record_into_step().
            Graph::StepRecording recording_;
    };

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Medium_StepRecorderDeathTest = Medium_StepRecorder;

} // namespace

/** @brief Tests that a list with no member reports empty and records zero tasks. */
TEST_F(Medium_StepRecorder, EmptyListRecordsNothing)
{
    StepRecorderList list;

    EXPECT_TRUE(list.empty());
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recording_));
    EXPECT_EQ(count, 0u);
}

/** @brief Tests that adding one recorder makes the list non-empty and calls nothing on it. */
TEST_F(Medium_StepRecorder, AddMakesTheListNonEmpty)
{
    InMemoryStepRecorder a(1);
    StepRecorderList list;

    SA_EXPECT_OK(list.add(a));

    EXPECT_FALSE(list.empty());
    EXPECT_TRUE(a.calls().empty());
}

/** @brief Tests that record_into_step returns the sum of the members' task counts. */
TEST_F(Medium_StepRecorder, RecordSumsTheMemberCounts)
{
    InMemoryStepRecorder a(2);
    InMemoryStepRecorder b(0);
    InMemoryStepRecorder c(5);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recording_));
    EXPECT_EQ(count, 7u);
}

/** @brief Tests that record_into_step calls each member once, in the order added. */
TEST_F(Medium_StepRecorder, RecordCallsMembersInTheOrderAdded)
{
    expect_record_order(recording_, {0, 1, 2}, {"a.record", "b.record", "c.record"});
    expect_record_order(recording_, {1, 0}, {"b.record", "a.record"});
}

/** @brief Tests that before_replay calls each member once, in the order added. */
TEST_F(Medium_StepRecorder, BeforeReplayCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    list.before_replay();

    EXPECT_EQ(journal, in_order_added("before"));
}

/** @brief Tests that after_replay calls each member once, in the order added. */
TEST_F(Medium_StepRecorder, AfterReplayCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    list.after_replay();

    EXPECT_EQ(journal, in_order_added("after"));
}

/** @brief Tests that a member with no replay hooks still records through the list. */
TEST_F(Medium_StepRecorder, MemberWithoutReplayHooksIsDriven)
{
    InMemoryStepRecorder a(4);
    RecordOnlyRecorder plain;
    StepRecorderList list;
    SA_EXPECT_OK(list.add(plain));
    SA_EXPECT_OK(list.add(a));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recording_));
    EXPECT_EQ(count, 7u);
    list.before_replay();
    list.after_replay();

    EXPECT_EQ(a.calls(), (std::vector<Call>{Call::Record, Call::BeforeReplay, Call::AfterReplay}));
}

/** @brief Tests that a list driven through a StepRecorder reference reaches its members. */
TEST_F(Medium_StepRecorder, ListIsDrivenThroughTheInterface)
{
    InMemoryStepRecorder a(2);
    InMemoryStepRecorder b(3);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    StepRecorder& trailing = list;

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, trailing.record_into_step(recording_));
    EXPECT_EQ(count, 5u);
    trailing.before_replay();
    trailing.after_replay();

    const std::vector<Call> expected = {Call::Record, Call::BeforeReplay, Call::AfterReplay};
    EXPECT_EQ(a.calls(), expected);
    EXPECT_EQ(b.calls(), expected);
}

/** @brief Tests that a list held as a member of another runs in its place in the outer order. */
TEST_F(Medium_StepRecorder, NestedListKeepsItsPlaceInTheOrder)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 2, journal);
    JournalRecorder c("c", 4, journal);
    StepRecorderList inner;
    SA_EXPECT_OK(inner.add(b));
    StepRecorderList outer;
    SA_EXPECT_OK(outer.add(a));
    SA_EXPECT_OK(outer.add(inner));
    SA_EXPECT_OK(outer.add(c));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, outer.record_into_step(recording_));
    EXPECT_EQ(count, 7u);
    EXPECT_EQ(journal, in_order_added("record"));
}

/** @brief Tests that adding a recorder the list already holds is fatal and leaves one entry. */
TEST_F(Medium_StepRecorderDeathTest, AddRejectsARecorderAlreadyHeld)
{
    InMemoryStepRecorder a(2);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));

    SA_EXPECT_FATAL((void)list.add(a), "StepRecorderList::add", "already");

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recording_));
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
