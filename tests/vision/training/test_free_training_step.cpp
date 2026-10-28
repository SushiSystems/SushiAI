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

    /** @brief Bounds the difference of two runs of one seed: none. */
    constexpr double K_SAME = 0.0;

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
} // namespace

/** @brief Tests that two runs of one seed end two steps with equal weights and terms. */
TEST_F(Medium_FreeTrainingStep, TwoRunsOfOneSeedEndWithEqualWeights)
{
    TwoLevelCase first;
    first.prepare(*engine);
    ASSERT_TRUE(first.ready);
    const std::vector<std::vector<double>> start = first.weights();
    std::vector<FreeLossTerms> first_terms;
    run_two_steps(first, *engine, first_terms);
    ASSERT_EQ(first_terms.size(), 2u);

    TwoLevelCase second;
    second.prepare(*engine);
    ASSERT_TRUE(second.ready);
    std::vector<FreeLossTerms> second_terms;
    run_two_steps(second, *engine, second_terms);
    ASSERT_EQ(second_terms.size(), 2u);

    const std::vector<std::vector<double>> left = first.weights();
    const std::vector<std::vector<double>> right = second.weights();
    ASSERT_EQ(left.size(), right.size());
    ASSERT_EQ(left.size(), start.size());

    double moved = 0.0;
    for (std::size_t entry = 0; entry < left.size(); ++entry)
    {
        ASSERT_EQ(left[entry].size(), right[entry].size()) << "entry " << entry;
        for (std::size_t i = 0; i < left[entry].size(); ++i)
        {
            EXPECT_NEAR(left[entry][i], right[entry][i], K_SAME)
                << "entry " << entry << " element " << i;
            moved = std::fmax(moved, std::fabs(left[entry][i] - start[entry][i]));
        }
    }
    EXPECT_GT(moved, K_MOVED);

    for (std::size_t run = 0; run < 2; ++run)
    {
        EXPECT_TRUE(std::isfinite(first_terms[run].total)) << "run " << run;
        EXPECT_GT(first_terms[run].total, 0.0) << "run " << run;
        EXPECT_NEAR(first_terms[run].total, second_terms[run].total, K_SAME) << "run " << run;
    }
    EXPECT_EQ(first.step->run_count(), 2u);
}

/** @brief Tests that the leaves hold what the stage assigned and the projection's bins. */
TEST_F(Medium_FreeTrainingStep, TheLeavesHoldTheStageAndTheProjection)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);

    const std::vector<double> sum = model.leaf_values("free.target_scores_sum");
    ASSERT_EQ(sum.size(), 1u);
    EXPECT_NEAR(sum[0],
                static_cast<double>(model.step->stage().targets().target_scores_sum[0]), K_SAME);
    EXPECT_GE(sum[0], 1.0);

    const std::vector<double> projection =
        model.leaf_values(Vision::Training::K_FREE_PROJECTION_ROLE);
    ASSERT_EQ(projection.size(), static_cast<std::size_t>(K_REG_MAX));
    EXPECT_NEAR(projection[0], 0.0, K_SAME);
    EXPECT_NEAR(projection[1], 1.0, K_SAME);
}

/** @brief Tests that a one-to-one copy trains every port and reads both branch weights. */
TEST_F(Medium_FreeTrainingStep, AOneToOneCopyMovesEveryWeightAndReadsBothBranchWeights)
{
    TwoLevelCase model;
    model.prepare(*engine, true);
    ASSERT_TRUE(model.ready);
    EXPECT_FLOAT_EQ(model.step->one_to_many_weight(), 0.8f);
    model.step->set_one_to_many_weight(0.25f);

    const std::vector<std::vector<double>> start = model.weights();
    std::vector<FreeLossTerms> terms;
    run_two_steps(model, *engine, terms);
    const std::vector<std::vector<double>> end = model.weights();
    ASSERT_EQ(start.size(), end.size());

    for (std::size_t entry = 0; entry < end.size(); ++entry)
    {
        double moved = 0.0;
        for (std::size_t i = 0; i < end[entry].size(); ++i)
            moved = std::fmax(moved, std::fabs(end[entry][i] - start[entry][i]));
        EXPECT_GT(moved, 0.0) << model.state->entries()[entry].name;
    }

    const std::vector<double> many =
        model.leaf_values(Vision::Training::K_FREE_ONE_TO_MANY_WEIGHT_ROLE);
    const std::vector<double> one =
        model.leaf_values(Vision::Training::K_FREE_ONE_TO_ONE_WEIGHT_ROLE);
    ASSERT_EQ(many.size(), 1u);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_NEAR(many[0], 0.25, K_SAME);
    EXPECT_NEAR(one[0], 0.75, K_SAME);
    EXPECT_GE(model.leaf_values("free.one_to_one.target_scores_sum")[0], 1.0);
    for (const FreeLossTerms& measured : terms)
        EXPECT_TRUE(std::isfinite(measured.total));
}

/** @brief Tests that a truth of another batch is refused before any replay. */
TEST_F(Medium_FreeTrainingStep, ATruthOfAnotherBatchIsRefused)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));

    GroundTruthBatch truth = make_truth();
    truth.batch = 1;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::Incompatible, "images");
    EXPECT_EQ(model.step->run_count(), 0u);
}

/** @brief Tests that a label outside the score ports' classes is refused by row. */
TEST_F(Medium_FreeTrainingStep, ALabelOutsideTheClassesIsRefused)
{
    TwoLevelCase model;
    model.prepare(*engine);
    ASSERT_TRUE(model.ready);
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor inputs, seeded_inputs(*engine));

    GroundTruthBatch truth = make_truth();
    truth.labels[1] = K_CLASSES;

    SA_EXPECT_FAILURE(model.step->run(inputs, truth), StatusCode::InvalidDescription, "row 1");
}

#if SA_ENABLE_PROFILER
/** @brief Tests that one step's host phases and device run fall in the period that opened it. */
TEST_F(Medium_FreeTrainingZones, AStepPutsReadbackAssignAndLossInItsPeriod)
{
    if (SA_PROFILE_LEVEL < 1)
        GTEST_SKIP() << "the build compiles every zone away";
    TwoLevelCase model;
    model.prepare(*engine);
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
