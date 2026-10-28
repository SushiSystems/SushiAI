/**************************************************************************/
/* test_zones.cpp                                                         */
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
 * @file test_zones.cpp
 * @brief Tests that Trainer::fit writes one period per step, holding the zones of its stages.
 * @author Mustafa Garip
 * @see tests/layers/train/test_trainer/test_zones.md
 */

#include <cstddef>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiProfiler/registry/registry.hpp>
#include <SushiProfiler/session/session.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/profile/profile_config.hpp>
#include <SushiAI/profile/timeline_binding.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile_records/profile_records.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_TrainerZones : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_CLASSES = 4;
    constexpr std::uint32_t K_STEPS = 2U;

} // namespace

TEST_F(Medium_TrainerZones, AFitOfTwoStepsWritesTwoPeriodsWithTheirGroups)
{
    if (SA_PROFILE_LEVEL < 1)
        GTEST_SKIP() << "the build compiles every zone away";
    MLP model(NN::Linear{K_FEATURES, 12}, NN::ReLU{}, NN::Linear{12, K_CLASSES});
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    SA_ASSERT_OK_AND_ASSIGN(auto optimizer, Optim::SGD::make(*engine, slots, sgd));
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(auto trainer, Train::Trainer::make(*engine, built, pool, objective,
                                                               *optimizer));
    Data::SyntheticOptions data_options;
    data_options.batch_size = K_BATCH;
    data_options.batches = K_STEPS;
    data_options.features = K_FEATURES;
    data_options.classes = K_CLASSES;
    SA_ASSERT_OK_AND_ASSIGN(auto dataset,
                            Data::SyntheticClassification::make(*engine, data_options));

    const SushiProfiler::Registry registry;
    SushiProfiler::SessionStart started = SushiProfiler::Session::start(
        SaTest::following_timeline("train.step", 0U, K_STEPS), registry);
    ASSERT_TRUE(started.ok()) << started.error;
    SushiProfiler::Session& session = *started.session;
    {
        const Profile::TimelineBinding binding(session);
        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        EXPECT_EQ(report.steps, static_cast<std::size_t>(K_STEPS));

        // The mark that would open a third step closes the second period.
        Profile::mark_period(Profile::Timeline::TrainStep);
    }
    const SushiProfiler::SessionStop stopped = session.stop();

    ASSERT_EQ(stopped.record.periods.size(), static_cast<std::size_t>(K_STEPS));
    const std::map<std::string, std::string> expected = {
        {"data.read", "data"},   {"host.feed", "host"},    {"copy.input", "copy"},
        {"device.run", "device"}, {"host.loss", "host"},   {"host.measure", "host"}};
    // SushiRuntime's own zones of the step's run land in the same period.
    const std::array<std::string_view, 1> runtime = {"Runtime"};
    for (const SushiProfiler::PeriodRecord& period : stopped.record.periods)
    {
        EXPECT_EQ(SaTest::sushiai_zones_of(stopped.record, period), expected)
            << "period " << period.index;
        EXPECT_FALSE(SaTest::zones_in_groups(stopped.record, period, runtime).empty())
            << "period " << period.index;
    }
}
