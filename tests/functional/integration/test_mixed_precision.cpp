/****************************************************************/
/* test_mixed_precision.cpp                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// The device half of AI-4. tests/functional/unit/test_precision.cpp pins the
// graph shape with no hardware anywhere; what is left for this file is the
// part that only a real Engine can answer:
//
//   - that `sycl::half` actually executes here, so the fp16 path is a computed
//     answer and not a plausible-looking graph nobody ran,
//   - that the *tensors* behind the graph carry the types the IR claims - a
//     master weight really is a 4-byte buffer and the copy a GEMM reads really
//     is a 2-byte one,
//   - and that the trainer refuses, before recording anything, a graph whose
//     dtypes disagree with the policy it was handed.
//
// Speed is deliberately not asserted anywhere. There is no low-precision
// matrix unit on this machine; a timing gate here would be pinning an artifact
// of emulation.

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
     * @brief AdamW at the rate `sa demo mlp` uses, not the library default.
     *
     * 1e-3 is the reference AdamW default and is right as a default; it is not
     * enough to move a hidden layer this small far in the couple of hundred
     * steps a test can afford, and a test whose "did it learn" threshold has
     * to be relaxed until it passes is not asserting anything. 3e-3 is the
     * demo's rate, so the two answer the same question at the same setting.
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
    // The claim this whole change is allowed to make on this hardware:
    // correctness. Not that it is faster - it is not, and cannot be here.
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

    // ln(4) is where a zeroed classifier starts; anything below half of it is
    // a model that has learned something rather than one that merely ran.
    EXPECT_NEAR(report.first_loss, std::log(static_cast<double>(CLASSES)), 1e-4);
    EXPECT_LT(report.last_loss, 0.5 * report.first_loss);
    EXPECT_GT(report.last_accuracy, 0.75);
}

TEST_F(Integration_MixedPrecision, TheMastersAreFourByteBuffersAndTheCopiesAreTwo)
{
    // The IR says FLOAT32 and HALF; this checks the tensors underneath say the
    // same, because the two could disagree - the pool materialises from the
    // value's description, and a planner that sized a slot from the wrong one
    // would hand out a view that overruns its neighbour.
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

    // The step ran at all, which is the loud half: SushiBLAS refuses a
    // mixed-dtype gemm by name, so a missing conversion could not get here.
    const Data::SyntheticClassification dataset(*engine, data_options());
    const Train::StepMetrics metrics = trainer.run_step(dataset.batch(0));
    EXPECT_TRUE(std::isfinite(metrics.loss));
}

TEST_F(Integration_MixedPrecision, TheTrainerRefusesAGraphItsPolicyDisowns)
{
    // The knob doing its job. A graph built without the cast pass and handed to
    // a trainer that was told it is mixed is exactly the mistake a caller makes
    // when they set one of the two knobs and forget the other - and without
    // this it is not an error, it is an fp32 run reported as a mixed one.
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
    // A defect this change would otherwise have introduced. score_accuracy()
    // reads the logits on the host and used to fall back to `data_as<float>()`
    // for anything that was not FLOAT64 - which on a half buffer reads two
    // elements as one and reports an accuracy that is fiction. Nothing throws,
    // nothing is NaN, and the run says 100%.
    //
    // build_classifier lists the logits as observed, so the shipped path cannot
    // reach this; a caller running the pass themselves can, and now finds out.
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
