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
 * @brief Tests that mixed precision keeps master weights in float32 and adds only conversions.
 * @author Mustafa Garip
 * @see tests/functional/regression/README.md
 */

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_status.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::MixedPrecisionPolicy;
using Graph::TensorPool;
using Graph::UniformPrecisionPolicy;
using Graph::ValueId;

namespace
{
    class Regression_MixedPrecision : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 16;
    constexpr int64_t K_FEATURES = 8;
    constexpr int64_t K_HIDDEN = 16;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 8;
    constexpr int64_t K_EPOCHS = 25;

    /**
     * @brief Returns AdamW options at the rate `sa demo mlp` uses, 3e-3.
     * @see tests/functional/regression/README.md
     */
    Optim::AdamWOptions optimizer_options()
    {
        Optim::AdamWOptions options;
        options.learning_rate = 3e-3;
        return options;
    }

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
     * @see tests/functional/regression/README.md
     */
    [[nodiscard]] Result<TrainingRun> train(SushiBLAS::Engine& engine,
                                            const Graph::PrecisionPolicy* precision)
    {
        Mlp model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::ReLU{}, NN::Linear{K_HIDDEN, K_CLASSES});

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
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(engine, built.graph, pool, built.spec, *optimizer,
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
                SA_TRY_ASSIGN(const Train::StepMetrics metrics,
                              trainer->run_step(dataset->batch(batch)));
                result.losses.push_back(metrics.loss);
                result.accuracy = metrics.accuracy;
            }

        // Dtypes are read after the run, not before; see tests/functional/regression/README.md.
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_MixedPrecision, TheMasterWeightsAndTheirGradientsStayFloat32)
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_MixedPrecision, TheFp32PolicyIsIndistinguishableFromNoPolicyAtAll)
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_MixedPrecision, NarrowingCostsConversionsAndNothingElseStructural)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(*engine, nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(*engine, &precision));

    EXPECT_EQ(mixed.operations, fp32.operations + mixed.casts)
        << "mixed precision changed the plan by more than the conversions it added";
}

/**
 * @brief Tests that mixed and single precision losses stay within 2e-2 relative at every step.
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_MixedPrecision, MixedAndSinglePrecisionTrainToTheSamePlace)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(*engine, nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(*engine, &precision));

    ASSERT_EQ(fp32.losses.size(), mixed.losses.size());
    ASSERT_GT(mixed.casts, 0u);

    // Both runs learned before they are compared; see tests/functional/regression/README.md.
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
 * @see tests/functional/regression/README.md
 */
TEST_F(Regression_MixedPrecision, AGeluLayerIsNotFoldedUnderANarrowingPolicy)
{
    using GeluMlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    GeluMlp wide_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{},
                       NN::Linear{K_HIDDEN, K_CLASSES});
    GeluMlp narrow_model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{},
                         NN::Linear{K_HIDDEN, K_CLASSES});

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
    EXPECT_EQ(wide.fusion.count(Graph::FusedForm::GemmBiasGelu), 1u)
        << wide.fusion.to_string();

    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GemmBiasGelu), 0u)
        << "a narrowing policy folded GELU into the GEMM, which puts the tanh "
           "approximation's x*x*x into fp16 - see graph/precision.hpp rule 5\n"
        << narrow.fusion.to_string();

    // The bias still folds on both layers; see tests/functional/regression/README.md.
    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GemmBias), 2u)
        << narrow.fusion.to_string();
    EXPECT_GT(narrow.casts.count(), 0u) << "this arm narrowed nothing, so it proves nothing";

    // The GELU stays in the graph, in the master type; see tests/functional/regression/README.md.
    for (const Graph::Node& node : narrow.graph.nodes())
        if (node.op == Ops::K_GELU.id)
            EXPECT_EQ(narrow.graph.value(node.outputs[0]).dtype, DType::Float32);
}
