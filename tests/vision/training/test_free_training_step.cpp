/**************************************************************************/
/* test_free_training_step.cpp                                            */
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
 * @file test_free_training_step.cpp
 * @brief Tests on an engine that a two-level anchor-free model trains on the leaves its
 *        host stage assigns, and that two runs of one seed end with equal weights.
 * @author Mustafa Garip
 */

#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <string>
#include <gtest/gtest.h>

#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiBLAS/engine/engine.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/profile/profile_config.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>

#include "free_two_level_case.hpp"

#if SA_ENABLE_PROFILER
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>
#include <SushiAI/profile/timeline_binding.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile_records/profile_records.hpp>
#endif

using namespace SushiAI;
using namespace SushiAI::FreeTwoLevelTest;
// SaTest stays reachable inside a TEST_F body; see tests/vision/training/README.md.
namespace SaTest = SushiAI::Test;
using Vision::Detect::GroundTruthBatch;
using Vision::Training::FreeLossTerms;

namespace
{
    class Medium_FreeTrainingStep : public SaTest::EngineTest {};

#if SA_ENABLE_PROFILER
    class Medium_FreeTrainingZones : public SaTest::EngineTest {};
#endif

    /** @brief Holds the least change of a weight that counts as having moved. */
    constexpr double K_MOVED = 1e-9;

    /** @brief Runs two steps of @p model on the seeded inputs and the fixture truth. */
    void run_two_steps(TwoLevelCase& model, SushiBLAS::Engine& engine,
                       std::vector<FreeLossTerms>& terms)
    {
        SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(engine));
        const GroundTruthBatch truth = make_truth();
        for (int run = 0; run < 2; ++run)
        {
            SA_ASSERT_OK_AND_ASSIGN(const FreeLossTerms measured, model.step->run(inputs, truth));
            terms.push_back(measured);
        }
    }

    /** @brief Expects every element of @p left to equal the same element of @p right. */
    void expect_weights_equal(const std::vector<std::vector<double>>& left,
                              const std::vector<std::vector<double>>& right)
    {
        ASSERT_EQ(left.size(), right.size());
        for (std::size_t entry = 0; entry < left.size(); ++entry)
            EXPECT_EQ(left[entry], right[entry]) << "entry " << entry;
    }

    /** @brief Returns the largest change of one element between @p start and @p end. */
    double largest_move(const std::vector<double>& start, const std::vector<double>& end)
    {
        double moved = 0.0;
        for (std::size_t i = 0; i < end.size(); ++i)
            moved = std::fmax(moved, std::fabs(end[i] - start[i]));

        return moved;
    }

    /** @brief Returns the names of the entries of @p model whose weights did not move. */
    std::vector<std::string> entries_that_stayed(const TwoLevelCase& model,
                                                 const std::vector<std::vector<double>>& start,
                                                 const std::vector<std::vector<double>>& end)
    {
        std::vector<std::string> stayed;
        for (std::size_t entry = 0; entry < end.size(); ++entry)
        {
            if (!(largest_move(start[entry], end[entry]) > 0.0))
                stayed.push_back(model.state->entries()[entry].name);
        }

        return stayed;
    }

    /** @brief Returns the terms of @p terms whose total is not finite, as run indices. */
    std::vector<std::size_t> runs_without_a_finite_total(const std::vector<FreeLossTerms>& terms)
    {
        std::vector<std::size_t> bad;
        for (std::size_t run = 0; run < terms.size(); ++run)
        {
            if (!std::isfinite(terms[run].total))
                bad.push_back(run);
        }

        return bad;
    }
} // namespace

/** @brief Tests that two runs of one seed end two steps with equal weights and terms. */
TEST_F(Medium_FreeTrainingStep, TwoRunsOfOneSeedEndWithEqualWeights)
{
    TwoLevelCase first;
    first.prepare(backend());
    ASSERT_TRUE(first.ready);
    const std::vector<std::vector<double>> start = first.weights();
    std::vector<FreeLossTerms> first_terms;
    run_two_steps(first, *engine, first_terms);
    ASSERT_EQ(first_terms.size(), 2u);

    TwoLevelCase second;
    second.prepare(backend());
    ASSERT_TRUE(second.ready);
    std::vector<FreeLossTerms> second_terms;
    run_two_steps(second, *engine, second_terms);
    ASSERT_EQ(second_terms.size(), 2u);

    const std::vector<std::vector<double>> left = first.weights();
    const std::vector<std::vector<double>> right = second.weights();
    ASSERT_EQ(left.size(), right.size());
    ASSERT_EQ(left.size(), start.size());

    expect_weights_equal(left, right);
    double moved = 0.0;
    for (std::size_t entry = 0; entry < left.size(); ++entry)
        moved = std::fmax(moved, largest_move(start[entry], left[entry]));
    EXPECT_GT(moved, K_MOVED);

    EXPECT_EQ(runs_without_a_finite_total(first_terms), std::vector<std::size_t>());
    EXPECT_GT(first_terms[0].total, 0.0);
    EXPECT_GT(first_terms[1].total, 0.0);
    EXPECT_EQ(first_terms[0].total, second_terms[0].total);
    EXPECT_EQ(first_terms[1].total, second_terms[1].total);
    EXPECT_EQ(first.step->run_count(), 2u);
}

/** @brief Tests that the sum leaf holds the stage's target score sum, at least 1. */
TEST_F(Medium_FreeTrainingStep, TheSumLeafHoldsTheStagesTargetSum)
{
    TwoLevelCase model;
    model.prepare(backend());
    ASSERT_TRUE(model.ready);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    const std::vector<double> sum = model.leaf_values("free.target_scores_sum");
    ASSERT_EQ(sum.size(), 1u);
    EXPECT_EQ(sum[0], static_cast<double>(model.step->stage().targets().target_scores_sum[0]));
    EXPECT_GE(sum[0], 1.0);
}

/** @brief Tests that the projection leaf holds the bin indices 0 and 1. */
TEST_F(Medium_FreeTrainingStep, TheProjectionLeafHoldsTheBinIndices)
{
    TwoLevelCase model;
    model.prepare(backend());
    ASSERT_TRUE(model.ready);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    EXPECT_EQ(model.leaf_values(Vision::Training::K_FREE_PROJECTION_ROLE),
              (std::vector<double>{0.0, 1.0}));
}

/** @brief Tests that a one-to-one copy trains every port. */
TEST_F(Medium_FreeTrainingStep, AOneToOneCopyMovesEveryWeight)
{
    TwoLevelCase model;
    model.prepare(backend(), true);
    ASSERT_TRUE(model.ready);
    const std::vector<std::vector<double>> start = model.weights();
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    EXPECT_EQ(entries_that_stayed(model, start, model.weights()), std::vector<std::string>());
}

/** @brief Tests that both branch weights are read back as the start and the set value. */
TEST_F(Medium_FreeTrainingStep, ABranchWeightIsReadBackFromItsLeaf)
{
    TwoLevelCase model;
    model.prepare(backend(), true);
    ASSERT_TRUE(model.ready);
    EXPECT_FLOAT_EQ(model.step->one_to_many_weight(), 0.8f);
    model.step->set_one_to_many_weight(0.25f);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    EXPECT_EQ(model.leaf_values(Vision::Training::K_FREE_ONE_TO_MANY_WEIGHT_ROLE),
              (std::vector<double>{0.25}));
    EXPECT_EQ(model.leaf_values(Vision::Training::K_FREE_ONE_TO_ONE_WEIGHT_ROLE),
              (std::vector<double>{0.75}));
    EXPECT_GE(model.leaf_values("free.one_to_one.target_scores_sum")[0], 1.0);
}

/** @brief Tests that the terms of a one-to-one run are finite. */
TEST_F(Medium_FreeTrainingStep, TheTermsOfAOneToOneRunAreFinite)
{
    TwoLevelCase model;
    model.prepare(backend(), true);
    ASSERT_TRUE(model.ready);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    EXPECT_EQ(runs_without_a_finite_total(terms), std::vector<std::size_t>());
}

/** @brief Tests that a refused truth counts no run; check_truth_layout owns the reason. */
TEST_F(Medium_FreeTrainingStep, ARefusedTruthCountsNoRun)
{
    TwoLevelCase model;
    model.prepare(backend());
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));

    GroundTruthBatch truth = make_truth();
    truth.batch = 1;

    EXPECT_FALSE(model.step->run(inputs, truth).has_value());
    EXPECT_EQ(model.step->run_count(), 0u);
}

/** @brief Tests that make refuses a pool that holds no tensor for a port, naming the port. */
TEST_F(Medium_FreeTrainingStep, MakeRefusesAPoolWithoutTheTensorsOfThePorts)
{
    TwoLevelCase model;
    model.prepare(backend());
    ASSERT_TRUE(model.ready);
    Graph::TensorPool empty(backend().engine());

    SA_EXPECT_FAILURE(Vision::Training::FreeTrainingStep::make(backend(), *model.objective,
                                                               model.built, empty,
                                                               *model.optimizer, {}),
                      StatusCode::InvalidDescription, "has no tensor in the training pool");
}

#if SA_ENABLE_PROFILER && SA_PROFILE_LEVEL >= 1
/** @brief Tests that one step's host phases and device run fall in the period that opened it. */
TEST_F(Medium_FreeTrainingZones, AStepPutsReadbackAssignAndLossInItsPeriod)
{
    TwoLevelCase model;
    model.prepare(backend());
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));
    const GroundTruthBatch truth = make_truth();

    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started = SushiProfiler::Session::start(
        SaTest::following_timeline("train.step", 0U, 1U), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    {
        const Profile::TimelineBinding binding(session);
        Profile::mark_period(Profile::Timeline::TrainStep);
        SA_EXPECT_OK(model.step->run(inputs, truth));

        // The next mark ends the period; a period with no end mark is not kept.
        Profile::mark_period(Profile::Timeline::TrainStep);
    }
    const SushiProfiler::SessionStop stopped = session.stop();

    ASSERT_EQ(stopped.record.periods.size(), 1u);
    const std::map<std::string, std::string> groups =
        SaTest::sushiai_zones_of(stopped.record, stopped.record.periods[0]);
    const std::map<std::string, std::string> expected = {
        {"copy.input", "copy"},     {"device.run", "device"}, {"host.assign", "host"},
        {"host.loss", "host"},      {"host.readback", "host"}};
    EXPECT_EQ(groups, expected);
}
#endif
