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
#include <SushiAI/graph/step_recorder.hpp>
#include "../../common/test_throws.hpp"

using namespace SushiAI;
using Graph::IStepRecorder;
using Graph::StepRecorderList;

namespace
{
    /** @brief Writes each call it receives to a shared journal as "<name>.<call>". */
    class JournalRecorder final : public IStepRecorder
    {
        public:
            /** @brief Binds the recorder to @p journal, which must outlive it. */
            JournalRecorder(std::string name, std::size_t tasks, std::vector<std::string>& journal)
                : name_(std::move(name)), tasks_(tasks), journal_(&journal)
            {

            }

            /** @brief Journals the call and returns the task count given at construction. */
            std::size_t record_into_step() override
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
    class RecordOnlyRecorder final : public IStepRecorder
    {
        public:
            /** @brief Returns a fixed count of three tasks. */
            std::size_t record_into_step() override
            {
                return 3;
            }
    };

    /** @brief Holds the journal entries three recorders named a, b and c write for @p call. */
    std::vector<std::string> in_order_added(const std::string& call)
    {
        return {"a." + call, "b." + call, "c." + call};
    }

} // namespace

/** @brief Tests that a list with no member reports empty, records zero tasks and replays. */
TEST(Unit_StepRecorder, EmptyListRecordsNothing)
{
    StepRecorderList list;

    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.record_into_step(), 0u);
    EXPECT_NO_THROW(list.before_replay());
    EXPECT_NO_THROW(list.after_replay());
}

/** @brief Tests that adding one recorder makes the list non-empty. */
TEST(Unit_StepRecorder, AddMakesTheListNonEmpty)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    StepRecorderList list;

    list.add(a);

    EXPECT_FALSE(list.empty());
    EXPECT_TRUE(journal.empty());
}

/** @brief Tests that record_into_step returns the sum of the members' task counts. */
TEST(Unit_StepRecorder, RecordSumsTheMemberCounts)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    JournalRecorder b("b", 0, journal);
    JournalRecorder c("c", 5, journal);
    StepRecorderList list;
    list.add(a);
    list.add(b);
    list.add(c);

    EXPECT_EQ(list.record_into_step(), 7u);
}

/** @brief Tests that record_into_step calls each member once, in the order added. */
TEST(Unit_StepRecorder, RecordCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    list.add(a);
    list.add(b);
    list.add(c);

    list.record_into_step();

    EXPECT_EQ(journal, in_order_added("record"));
}

/** @brief Tests that the order of the calls follows add, not the order of construction. */
TEST(Unit_StepRecorder, OrderFollowsAddNotConstruction)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    StepRecorderList list;
    list.add(b);
    list.add(a);

    list.record_into_step();

    EXPECT_EQ(journal, (std::vector<std::string>{"b.record", "a.record"}));
}

/** @brief Tests that before_replay calls each member once, in the order added. */
TEST(Unit_StepRecorder, BeforeReplayCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    list.add(a);
    list.add(b);
    list.add(c);

    list.before_replay();

    EXPECT_EQ(journal, in_order_added("before"));
}

/** @brief Tests that after_replay calls each member once, in the order added. */
TEST(Unit_StepRecorder, AfterReplayCallsMembersInTheOrderAdded)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 1, journal);
    JournalRecorder c("c", 1, journal);
    StepRecorderList list;
    list.add(a);
    list.add(b);
    list.add(c);

    list.after_replay();

    EXPECT_EQ(journal, in_order_added("after"));
}

/** @brief Tests that a member with no replay hooks still records through the list. */
TEST(Unit_StepRecorder, MemberWithoutReplayHooksIsDriven)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 4, journal);
    RecordOnlyRecorder plain;
    StepRecorderList list;
    list.add(plain);
    list.add(a);

    EXPECT_EQ(list.record_into_step(), 7u);
    list.before_replay();
    list.after_replay();

    EXPECT_EQ(journal, (std::vector<std::string>{"a.record", "a.before", "a.after"}));
}

/** @brief Tests that a list driven through an IStepRecorder pointer reaches its members. */
TEST(Unit_StepRecorder, ListIsDrivenThroughTheInterface)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    JournalRecorder b("b", 3, journal);
    StepRecorderList list;
    list.add(a);
    list.add(b);
    IStepRecorder* trailing = &list;

    EXPECT_EQ(trailing->record_into_step(), 5u);
    trailing->before_replay();
    trailing->after_replay();

    EXPECT_EQ(journal, (std::vector<std::string>{"a.record", "b.record", "a.before", "b.before",
                                                 "a.after", "b.after"}));
}

/** @brief Tests that a list held as a member of another runs in its place in the outer order. */
TEST(Unit_StepRecorder, NestedListKeepsItsPlaceInTheOrder)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 1, journal);
    JournalRecorder b("b", 2, journal);
    JournalRecorder c("c", 4, journal);
    StepRecorderList inner;
    inner.add(b);
    StepRecorderList outer;
    outer.add(a);
    outer.add(inner);
    outer.add(c);

    EXPECT_EQ(outer.record_into_step(), 7u);
    EXPECT_EQ(journal, in_order_added("record"));
}

/** @brief Tests that adding a recorder the list already holds fails and leaves one entry. */
TEST(Unit_StepRecorder, AddRejectsARecorderAlreadyHeld)
{
    std::vector<std::string> journal;
    JournalRecorder a("a", 2, journal);
    StepRecorderList list;
    list.add(a);

    SA_EXPECT_ERROR(list.add(a), "StepRecorderList::add", "already");

    EXPECT_EQ(list.record_into_step(), 2u);
    EXPECT_EQ(journal, (std::vector<std::string>{"a.record"}));
}

/** @brief Tests that adding a list to itself fails and leaves the list empty. */
TEST(Unit_StepRecorder, AddRejectsTheListItself)
{
    StepRecorderList list;

    SA_EXPECT_ERROR(list.add(list), "StepRecorderList::add", "itself");

    EXPECT_TRUE(list.empty());
}
