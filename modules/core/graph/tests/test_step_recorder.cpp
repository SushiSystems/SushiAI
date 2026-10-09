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
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiRuntime/graph/task_graph.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <common/test_environment.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Graph::StepRecorder;
using Graph::StepRecorderList;

namespace
{
    /** @brief Writes each call it receives to a shared journal as "<name>.<call>". */
    class JournalRecorder final : public StepRecorder
    {
        public:
            /** @brief Binds the recorder to @p journal, which must outlive it. */
            JournalRecorder(std::string name, std::size_t tasks, std::vector<std::string>& journal)
                : name_(std::move(name)), tasks_(tasks), journal_(&journal)
            {

            }

            /** @brief Journals the call and returns the task count given at construction. */
            [[nodiscard]] Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder&) override
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
            [[nodiscard]] Result<std::size_t> record_into_step(SushiBLAS::TaskRecorder&) override
            {
                return 3;
            }
    };

    /** @brief Holds the journal entries three recorders named a, b and c write for @p call. */
    std::vector<std::string> in_order_added(const std::string& call)
    {
        return {"a." + call, "b." + call, "c." + call};
    }

    /** @brief Gives each test a task graph and a recorder over it, on the shared context. */
    class Unit_StepRecorder : public ::testing::Test
    {
        protected:
            /** @brief Builds the graph, its scratch pool and the recorder the tests use. */
            Unit_StepRecorder()
                : graph_(SushiAI::Test::TestEnvironment::context()),
                  scratch_(SushiAI::Test::TestEnvironment::context()),
                  recorder_(graph_, scratch_)
            {

            }

            /// @brief Task graph the recorder writes to; no test compiles it.
            SushiRuntime::Graph::TaskGraph graph_;

            /// @brief Scratch pool the recorder leases from.
            SushiBLAS::ScratchPool scratch_;

            /// @brief Recorder every test passes to record_into_step().
            SushiBLAS::TaskRecorder recorder_;
    };

    /** @brief Names the suite of the cases that expect the process to stop. */
    using Unit_StepRecorderDeathTest = Unit_StepRecorder;

} // namespace

/** @brief Tests that a list with no member reports empty, records zero tasks and replays. */
TEST_F(Unit_StepRecorder, EmptyListRecordsNothing)
{
    StepRecorderList list;

    EXPECT_TRUE(list.empty());
    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recorder_));
    EXPECT_EQ(count, 0u);
    EXPECT_NO_THROW(list.before_replay());
    EXPECT_NO_THROW(list.after_replay());
}

/** @brief Tests that adding one recorder makes the list non-empty. */
TEST_F(Unit_StepRecorder, AddMakesTheListNonEmpty)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    StepRecorderList list;

    SA_EXPECT_OK(list.add(a));

    EXPECT_FALSE(list.empty());
    EXPECT_TRUE(journal.empty());
}

/** @brief Tests that record_into_step returns the sum of the members' task counts. */
TEST_F(Unit_StepRecorder, RecordSumsTheMemberCounts)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    JournalRecorder b("b", 0, journal);
    JournalRecorder c("c", 5, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recorder_));
    EXPECT_EQ(count, 7u);
}

/** @brief Tests that record_into_step calls each member once, in the order added. */
TEST_F(Unit_StepRecorder, RecordCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(c));

    SA_EXPECT_OK(list.record_into_step(recorder_));

    EXPECT_EQ(journal, in_order_added("record"));
}

/** @brief Tests that the order of the calls follows add, not the order of construction. */
TEST_F(Unit_StepRecorder, OrderFollowsAddNotConstruction)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(b));
    SA_EXPECT_OK(list.add(a));

    SA_EXPECT_OK(list.record_into_step(recorder_));

    EXPECT_EQ(journal, (std::vector<std::string>{"b.record", "a.record"}));
}

/** @brief Tests that before_replay calls each member once, in the order added. */
TEST_F(Unit_StepRecorder, BeforeReplayCallsMembersInTheOrderAdded)
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
TEST_F(Unit_StepRecorder, AfterReplayCallsMembersInTheOrderAdded)
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
TEST_F(Unit_StepRecorder, MemberWithoutReplayHooksIsDriven)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 4, journal);
    RecordOnlyRecorder plain;
    StepRecorderList list;
    SA_EXPECT_OK(list.add(plain));
    SA_EXPECT_OK(list.add(a));

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recorder_));
    EXPECT_EQ(count, 7u);
    list.before_replay();
    list.after_replay();

    EXPECT_EQ(journal, (std::vector<std::string>{"a.record", "a.before", "a.after"}));
}

/** @brief Tests that a list driven through an IStepRecorder pointer reaches its members. */
TEST_F(Unit_StepRecorder, ListIsDrivenThroughTheInterface)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    JournalRecorder b("b", 3, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));
    SA_EXPECT_OK(list.add(b));
    StepRecorder* trailing = &list;

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, trailing->record_into_step(recorder_));
    EXPECT_EQ(count, 5u);
    trailing->before_replay();
    trailing->after_replay();

    EXPECT_EQ(journal, (std::vector<std::string>{"a.record", "b.record", "a.before", "b.before",
                                                 "a.after", "b.after"}));
}

/** @brief Tests that a list held as a member of another runs in its place in the outer order. */
TEST_F(Unit_StepRecorder, NestedListKeepsItsPlaceInTheOrder)
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

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, outer.record_into_step(recorder_));
    EXPECT_EQ(count, 7u);
    EXPECT_EQ(journal, in_order_added("record"));
}

/** @brief Tests that adding a recorder the list already holds is fatal and leaves one entry. */
TEST_F(Unit_StepRecorderDeathTest, AddRejectsARecorderAlreadyHeld)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    StepRecorderList list;
    SA_EXPECT_OK(list.add(a));

    SA_EXPECT_FATAL((void)list.add(a), "StepRecorderList::add", "already");

    SA_ASSERT_OK_AND_ASSIGN(const std::size_t count, list.record_into_step(recorder_));
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(journal, (std::vector<std::string>{"a.record"}));
}

/** @brief Tests that adding a list to itself is fatal and leaves the list empty. */
TEST_F(Unit_StepRecorderDeathTest, AddRejectsTheListItself)
{
    StepRecorderList list;

    SA_EXPECT_FATAL((void)list.add(list), "StepRecorderList::add", "itself");

    EXPECT_TRUE(list.empty());
}
