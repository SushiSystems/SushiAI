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
 * @brief Tests the device half of mixed precision: execution, tensor dtypes and the policy check.
 * @author Mustafa Garip
 */

#include <cmath>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../common/test_utils.hpp"

using namespace SushiAI;

namespace SaTest = SushiAI::Test;
using Graph::CastReport;
using Graph::InsertedCast;
using Graph::MixedPrecisionPolicy;
using Graph::TensorPool;
using Graph::UniformPrecisionPolicy;
using Graph::ValueId;

namespace
{
    class Integration_MixedPrecision : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    Mlp make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return Mlp(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    constexpr int64_t BATCH = 16;
    constexpr int64_t FEATURES = 8;
    constexpr int64_t HIDDEN = 12;
    constexpr int64_t CLASSES = 4;

    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = BATCH;
        options.batches = 8;
        options.features = FEATURES;
        options.classes = CLASSES;
        return options;
    }

    /**
     * @brief Returns AdamW options at the 3e-3 rate `sa demo mlp` uses; see
     *        tests/functional/integration/README.md.
     */
    Optim::AdamWOptions optimizer_options()
    {
        Optim::AdamWOptions options;
        options.learning_rate = 3e-3;
        return options;
    }
}

TEST_F(Integration_MixedPrecision, AModelWithFp16ProductsStillLearns)
{
    // Correctness is asserted; speed is not; see tests/functional/integration/README.md.
    Mlp model = make_model(FEATURES, HIDDEN, CLASSES);

    const MixedPrecisionPolicy precision;
    Train::TrainingGraph built = Train::build_classifier(
        model, BATCH, FEATURES, CLASSES, DType::FLOAT32, Graph::FusionOptions{}, &precision);

    ASSERT_GT(built.casts.count(), 0u) << "the mixed policy inserted nothing";

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamW optimizer(*engine, slots, optimizer_options());

    Train::TrainerOptions options;
    options.epochs = 25;
    options.precision = &precision;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    const Data::SyntheticClassification dataset(*engine, data_options());
    const Train::RunReport report = trainer.fit(dataset);

    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_TRUE(std::isfinite(report.first_loss));
    EXPECT_TRUE(std::isfinite(report.last_loss));

    // ln(4) is where a zeroed classifier starts; see tests/functional/integration/README.md.
    EXPECT_NEAR(report.first_loss, std::log(static_cast<double>(CLASSES)), 1e-4);
    EXPECT_LT(report.last_loss, 0.5 * report.first_loss);
    EXPECT_GT(report.last_accuracy, 0.75);
}

TEST_F(Integration_MixedPrecision, TheMastersAreFourByteBuffersAndTheCopiesAreTwo)
{
    // Tensors must carry the dtypes the IR claims; see tests/functional/integration/README.md.
    Mlp model = make_model(FEATURES, HIDDEN, CLASSES);

    const MixedPrecisionPolicy precision;
    Train::TrainingGraph built = Train::build_classifier(
        model, BATCH, FEATURES, CLASSES, DType::FLOAT32, Graph::FusionOptions{}, &precision);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamW optimizer(*engine, slots, optimizer_options());

    Train::TrainerOptions options;
    options.precision = &precision;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    // Every tensor the optimizer touches - both sides of the update.
    for (const Optim::ParameterSlot& slot : slots)
    {
        EXPECT_EQ(slot.parameter.dtype, DType::FLOAT32) << slot.name << " master weight";
        EXPECT_EQ(slot.gradient.dtype, DType::FLOAT32) << slot.name << " gradient";
    }

    // And at least one genuine half buffer exists, so the fp16 path is real.
    std::size_t half_buffers = 0;
    for (const InsertedCast& cast : built.casts.casts())
        if (cast.to == DType::HALF)
        {
            ASSERT_TRUE(pool.is_bound(cast.destination))
                << "cast destination " << cast.destination << " was never materialised";
            EXPECT_EQ(pool.tensor(cast.destination).dtype, DType::HALF);
            ++half_buffers;
        }

    EXPECT_GT(half_buffers, 0u);

    // SushiBLAS refuses a mixed-dtype gemm by name; see tests/functional/integration/README.md.
    const Data::SyntheticClassification dataset(*engine, data_options());
    const Train::StepMetrics metrics = trainer.run_step(dataset.batch(0));
    EXPECT_TRUE(std::isfinite(metrics.loss));
}

TEST_F(Integration_MixedPrecision, TheTrainerRefusesAGraphItsPolicyDisowns)
{
    // The graph skipped the cast pass on purpose; see tests/functional/integration/README.md.
    Mlp model = make_model(FEATURES, HIDDEN, CLASSES);
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamW optimizer(*engine, slots, optimizer_options());

    const MixedPrecisionPolicy precision;
    Train::TrainerOptions options;
    options.precision = &precision;

    EXPECT_THROW(
        Train::Trainer(*engine, built.graph, pool, built.spec, optimizer, options),
        Error);

    // The same graph with the policy it was actually built under is accepted.
    const UniformPrecisionPolicy single;
    Train::TrainerOptions consistent;
    consistent.precision = &single;
    EXPECT_NO_THROW(
        Train::Trainer(*engine, built.graph, pool, built.spec, optimizer, consistent));
}

TEST_F(Integration_MixedPrecision, AccuracyRefusesLogitsItCannotRead)
{
    // A half buffer must not be read as float; see tests/functional/integration/README.md.
    Mlp model = make_model(FEATURES, HIDDEN, CLASSES);
    Train::TrainingGraph built = Train::build_classifier(model, BATCH, FEATURES, CLASSES);

    // Deliberately narrow the logits: no observed list, so nothing pins them.
    const MixedPrecisionPolicy precision;
    Graph::CastOptions casts;
    casts.policy = &precision;
    Graph::insert_casts(built.graph, casts);

    ASSERT_EQ(built.graph.value(built.spec.logits).dtype, DType::HALF);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, model, pool);

    Optim::AdamW optimizer(*engine, slots, optimizer_options());

    Train::TrainerOptions options;
    options.track_accuracy = true;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    const Data::SyntheticClassification dataset(*engine, data_options());
    EXPECT_THROW(trainer.run_step(dataset.batch(0)), Error);
}
