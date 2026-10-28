/**************************************************************************/
/* test_mixed_precision.cpp                                               */
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
 * @file test_mixed_precision.cpp
 * @brief Tests mixed precision: execution, tensor dtypes, the policy check and the float32 master
 *        weights.
 * @author Mustafa Garip
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;
using Graph::CastReport;
using Graph::InsertedCast;
using Graph::MixedPrecisionPolicy;
using Graph::TensorPool;
using Graph::UniformPrecisionPolicy;
using Graph::ValueID;

namespace
{
    class Medium_MixedPrecision : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return MLP(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    constexpr int64_t K_BATCH = 16;
    constexpr int64_t K_FEATURES = 8;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;

    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = 8;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        return options;
    }

    /**
     * @brief Returns AdamW options at the 3e-3 rate `sa demo mlp` uses; see
     *        tests/layers/train/README.md.
     */
    Optim::AdamWOptions optimizer_options()
    {
        Optim::AdamWOptions options;
        options.learning_rate = 3e-3;
        return options;
    }
}

TEST_F(Medium_MixedPrecision, AModelWithFP16ProductsStillLearns)
{
    // Correctness is asserted; speed is not; see tests/layers/train/README.md.
    MLP model = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);

    const MixedPrecisionPolicy precision;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES,
                                                    DType::Float32, Graph::FusionOptions{},
                                                    &precision));

    ASSERT_GT(built.casts.count(), 0u) << "the mixed policy inserted nothing";

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options()));

    Train::TrainerOptions options;
    options.epochs = 25;
    options.precision = &precision;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options()));
    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_TRUE(std::isfinite(report.first_loss));
    EXPECT_TRUE(std::isfinite(report.last_loss));

    // ln(4) is where a zeroed classifier starts; see tests/layers/train/README.md.
    EXPECT_NEAR(report.first_loss, std::log(static_cast<double>(K_CLASSES)), 1e-4);
    EXPECT_LT(report.last_loss, 0.5 * report.first_loss);
    ASSERT_NE(report.last_metrics.find("accuracy"), nullptr);
    EXPECT_GT(report.last_metrics.find("accuracy")->value, 0.75);
}

TEST_F(Medium_MixedPrecision, TheMastersAreFourByteBuffersAndTheCopiesAreTwo)
{
    // Tensors must carry the dtypes the IR claims; see tests/layers/train/README.md.
    MLP model = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);

    const MixedPrecisionPolicy precision;
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES,
                                                    DType::Float32, Graph::FusionOptions{},
                                                    &precision));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options()));

    Train::TrainerOptions options;
    options.precision = &precision;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer, options));

    // Every tensor the optimizer touches - both sides of the update.
    for (const Optim::ParameterSlot& slot : slots)
    {
        EXPECT_EQ(slot.parameter.dtype, DType::Float32) << slot.name << " master weight";
        EXPECT_EQ(slot.gradient.dtype, DType::Float32) << slot.name << " gradient";
    }

    // And at least one genuine half buffer exists, so the fp16 path is real.
    std::size_t half_buffers = 0;
    for (const InsertedCast& cast : built.casts.casts())
        if (cast.to == DType::Half)
        {
            ASSERT_TRUE(pool.is_bound(cast.destination))
                << "cast destination " << cast.destination << " was never materialised";
            EXPECT_EQ(pool.tensor(cast.destination).dtype, DType::Half);
            ++half_buffers;
        }

    EXPECT_GT(half_buffers, 0u);

    // SushiBLAS refuses a mixed-dtype gemm by name; see tests/layers/train/README.md.
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options()));
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                            trainer->run_step(dataset_batch));
    EXPECT_TRUE(std::isfinite(metrics.loss));
}

TEST_F(Medium_MixedPrecision, TheTrainerRefusesAGraphItsPolicyDisowns)
{
    // The graph skipped the cast pass on purpose; see tests/layers/train/README.md.
    MLP model = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options()));

    const MixedPrecisionPolicy precision;
    Train::TrainerOptions options;
    options.precision = &precision;

    const auto objective = SaTest::classifier_objective(built);
    SA_EXPECT_FAILURE(Train::Trainer::make(*engine, built, pool, objective, *optimizer,
                                           options),
                      StatusCode::InvalidDescription, "does not agree with the precision policy");

    // The same graph with the policy it was actually built under is accepted.
    const UniformPrecisionPolicy single;
    Train::TrainerOptions consistent;
    consistent.precision = &single;
    SA_EXPECT_OK(Train::Trainer::make(*engine, built, pool, objective, *optimizer,
                                      consistent));
}

TEST_F(Medium_MixedPrecision, AccuracyRefusesLogitsItCannotRead)
{
    // A half buffer must not be read as float; see tests/layers/train/README.md.
    MLP model = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    // Deliberately narrow the logits: no observed list, so nothing pins them.
    const MixedPrecisionPolicy precision;
    Graph::CastOptions casts;
    casts.policy = &precision;
    SA_EXPECT_OK(Graph::insert_casts(built.graph, casts));

    ASSERT_EQ(built.graph.value(built.spec.logits).dtype, DType::Half);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(built, state, pool));
    SA_EXPECT_OK(NN::initialize(*engine, state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                            Optim::AdamW::make(*engine, slots, optimizer_options()));

    Train::TrainerOptions options;
    options.track_metrics = true;
    const auto objective = SaTest::classifier_objective(built);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                            Train::Trainer::make(*engine, built, pool, objective,
                                                 *optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, data_options()));
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_EXPECT_FAILURE(trainer->run_step(dataset_batch), StatusCode::InvalidDescription,
                      "needs a host read of the logits");
}

namespace
{
    class Medium_MixedPrecisionRegressions : public SaTest::EngineTest {};

    constexpr int64_t K_GRAPH_HIDDEN = 16;
    constexpr int64_t K_BATCHES = 8;
    constexpr int64_t K_EPOCHS = 25;

    /**
     * @struct TrainingRun
     * @brief One whole training run's observable result.
     */
    struct TrainingRun
    {
        std::vector<double> losses;      /**< One per step, in order. */
        std::size_t casts = 0;           /**< How many conversions the pass added. */
        std::size_t operations = 0;      /**< Recorded SushiBLAS calls per step. */
        std::vector<DType> master_dtypes; /**< Every parameter tensor's element type. */
        std::vector<DType> gradient_dtypes;
        double accuracy = 0.0;
    };

    /**
     * @brief Trains the same model, data and seed under @p precision.
     * @see tests/layers/train/README.md
     */
    [[nodiscard]] Result<TrainingRun> train(SushiBLAS::Engine& engine,
                                            const Graph::PrecisionPolicy* precision)
    {
        MLP model(NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::ReLU{}, NN::Linear{K_GRAPH_HIDDEN, K_CLASSES});

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float32,
                                              Graph::FusionOptions{}, precision));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(built, state, pool));
        SA_TRY(NN::initialize(engine, state));

        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(engine, slots, optimizer_options()));

        Train::TrainerOptions options;
        options.precision = precision;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built, pool, objective, *optimizer,
                                           options));

        Data::SyntheticOptions data;
        data.batch_size = K_BATCH;
        data.batches = K_BATCHES;
        data.features = K_FEATURES;
        data.classes = K_CLASSES;
        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(engine, data));

        TrainingRun result;
        result.casts = built.casts.count();
        result.operations = trainer->step().operation_count();

        for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                SA_TRY(dataset->read(batch, dataset_batch));
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset_batch));
                result.losses.push_back(metrics.loss);
                result.accuracy = metrics.metrics.value_or("accuracy", 0.0);
            }

        // Dtypes are read after the run, not before; see tests/layers/train/README.md.
        for (const Optim::ParameterSlot& slot : slots)
        {
            result.master_dtypes.push_back(slot.parameter.dtype);
            result.gradient_dtypes.push_back(slot.gradient.dtype);
        }

        return result;
    }
}

/**
 * @brief Tests that every master weight and gradient is float32 after a mixed-precision run.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, TheMasterWeightsAndTheirGradientsStayFloat32)
{
    const MixedPrecisionPolicy precision;
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(*engine, &precision));

    ASSERT_FALSE(mixed.master_dtypes.empty());
    ASSERT_GT(mixed.casts, 0u) << "nothing was narrowed, so nothing was being guarded";

    for (DType dtype : mixed.master_dtypes)
        EXPECT_EQ(dtype, DType::Float32)
            << "a master weight was narrowed; fp16 updates below the weight's own "
               "spacing are discarded and the run still converges";

    for (DType dtype : mixed.gradient_dtypes)
        EXPECT_EQ(dtype, DType::Float32)
            << "a gradient reached the optimizer in low precision";
}

/**
 * @brief Tests that the fp32 policy adds no cast and matches no policy bit for bit.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, TheFP32PolicyIsIndistinguishableFromNoPolicyAtAll)
{
    const UniformPrecisionPolicy single;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun none, train(*engine, nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun explicit_fp32, train(*engine, &single));

    EXPECT_EQ(none.casts, 0u);
    EXPECT_EQ(explicit_fp32.casts, 0u);
    EXPECT_EQ(none.operations, explicit_fp32.operations);

    ASSERT_EQ(none.losses.size(), explicit_fp32.losses.size());
    for (std::size_t step = 0; step < none.losses.size(); ++step)
        EXPECT_EQ(none.losses[step], explicit_fp32.losses[step]) << "step " << step;
}

/**
 * @brief Tests that mixed precision adds exactly as many operations as casts.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, NarrowingCostsConversionsAndNothingElseStructural)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(*engine, nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(*engine, &precision));

    EXPECT_EQ(mixed.operations, fp32.operations + mixed.casts)
        << "mixed precision changed the plan by more than the conversions it added";
}

/**
 * @brief Tests that mixed and single precision losses stay within 2e-2 relative at every step.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, MixedAndSinglePrecisionTrainToTheSamePlace)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(*engine, nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(*engine, &precision));

    ASSERT_EQ(fp32.losses.size(), mixed.losses.size());
    ASSERT_GT(mixed.casts, 0u);

    // Both runs learned before they are compared; see tests/layers/train/README.md.
    EXPECT_LT(fp32.losses.back(), 0.5 * fp32.losses.front());
    EXPECT_LT(mixed.losses.back(), 0.5 * mixed.losses.front());
    EXPECT_GT(mixed.accuracy, 0.75);

    for (std::size_t step = 0; step < fp32.losses.size(); ++step)
    {
        ASSERT_TRUE(std::isfinite(mixed.losses[step])) << "step " << step;

        const double scale = std::max(std::abs(fp32.losses[step]), 1e-6);
        EXPECT_LT(std::abs(mixed.losses[step] - fp32.losses[step]) / scale, 2e-2)
            << "step " << step << ": fp32 " << fp32.losses[step]
            << " vs mixed " << mixed.losses[step];
    }
}

/**
 * @brief Tests that a narrowing policy leaves GELU unfolded and in float32.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, AGELULayerIsNotFoldedUnderANarrowingPolicy)
{
    using GELUMLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    GELUMLP wide_model(NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::GELU{},
                       NN::Linear{K_GRAPH_HIDDEN, K_CLASSES});
    GELUMLP narrow_model(NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::GELU{},
                         NN::Linear{K_GRAPH_HIDDEN, K_CLASSES});

    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph wide,
                            Train::build_classifier(wide_model, K_BATCH, K_FEATURES, K_CLASSES,
                                                    DType::Float32, Graph::FusionOptions{},
                                                    nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph narrow,
                            Train::build_classifier(narrow_model, K_BATCH, K_FEATURES, K_CLASSES,
                                                    DType::Float32, Graph::FusionOptions{},
                                                    &precision));

    // The control: with nothing narrowing, the entry is reachable and taken.
    EXPECT_EQ(wide.fusion.count(Graph::FusedForm::GEMMBiasGELU), 1u)
        << wide.fusion.to_string();

    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMMBiasGELU), 0u)
        << "a narrowing policy folded GELU into the GEMM, which puts the tanh "
           "approximation's x*x*x into fp16 - see graph/precision.hpp rule 5\n"
        << narrow.fusion.to_string();

    // The bias still folds on both layers; see tests/layers/train/README.md.
    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMMBias), 2u)
        << narrow.fusion.to_string();
    EXPECT_GT(narrow.casts.count(), 0u) << "this arm narrowed nothing, so it proves nothing";

    // The GELU stays in the graph, in the master type; see tests/layers/train/README.md.
    for (const Graph::Node& node : narrow.graph.nodes())
        if (node.op == Ops::K_GELU.id)
            EXPECT_EQ(narrow.graph.value(node.outputs[0]).dtype, DType::Float32);
}
