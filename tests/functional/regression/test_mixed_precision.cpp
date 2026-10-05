/****************************************************************/
/* test_mixed_precision.cpp                                     */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

// Every guarantee here fails *silently*. That is what puts them in this tier
// rather than beside the integration tests: none of them can be caught by a
// run that reports success.
//
//   - A master weight retyped to fp16 does not throw. The optimizer keeps
//     working, the loss keeps falling, and updates smaller than the weight's
//     own fp16 spacing are simply discarded - which is the failure master
//     weights exist to prevent and looks exactly like a learning rate that
//     wants tuning.
//   - A conversion placed on the wrong side of a reduction does not throw
//     either; the model just learns worse.
//   - And the control matters as much as the case: if the fp32 policy also
//     inserted casts, every "mixed inserted N casts" assertion anywhere in
//     this repository would be measuring the pass's unconditional behaviour
//     rather than the policy's decision.

#include <cmath>
#include <string>
#include <vector>
#include <gtest/gtest.h>
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

    constexpr int64_t BATCH = 16;
    constexpr int64_t FEATURES = 8;
    constexpr int64_t HIDDEN = 16;
    constexpr int64_t CLASSES = 4;
    constexpr int64_t BATCHES = 8;
    constexpr int64_t EPOCHS = 25;

    /**
     * @brief AdamW at the rate `sa demo mlp` uses, not the library default.
     *
     * 1e-3 is the reference AdamW default; it is not enough to move a model
     * this small far in the couple of hundred steps a test can afford, and a
     * comparison between two runs that both barely moved would agree for the
     * wrong reason. 3e-3 is the demo's rate, so the numbers here and the ones
     * in docs/README.md come from the same setting.
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
     *
     * Everything except the policy is fixed, so a difference between two runs
     * is the policy and nothing else: the initialiser and the dataset are both
     * seeded, and the seeds are the library defaults rather than a value this
     * file picked, so a change to either shows up as a changed trajectory in
     * both arms rather than as a silent divergence in one.
     */
    TrainingRun train(SushiBLAS::Engine& engine, const Graph::IPrecisionPolicy* precision)
    {
        Mlp model(NN::Linear{FEATURES, HIDDEN}, NN::ReLU{}, NN::Linear{HIDDEN, CLASSES});

        Train::TrainingGraph built = Train::build_classifier(
            model, BATCH, FEATURES, CLASSES, DType::FLOAT32, Graph::FusionOptions{}, precision);

        TensorPool pool(engine);
        std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
        NN::initialize(engine, model, pool);

        Optim::AdamW optimizer(engine, slots, optimizer_options());

        Train::TrainerOptions options;
        options.precision = precision;
        Train::Trainer trainer(engine, built.graph, pool, built.spec, optimizer, options);

        Data::SyntheticOptions data;
        data.batch_size = BATCH;
        data.batches = BATCHES;
        data.features = FEATURES;
        data.classes = CLASSES;
        const Data::SyntheticClassification dataset(engine, data);

        TrainingRun result;
        result.casts = built.casts.count();
        result.operations = trainer.step().operation_count();

        for (int64_t epoch = 0; epoch < EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset.batch_count(); ++batch)
            {
                const Train::StepMetrics metrics = trainer.run_step(dataset.batch(batch));
                result.losses.push_back(metrics.loss);
                result.accuracy = metrics.accuracy;
            }

        // Read *after* the run, not before: fusion and the cast pass both
        // rewrite nodes, and a master weight that survived the passes could
        // still have been swapped for a narrow copy by the optimizer wiring.
        for (const Optim::ParameterSlot& slot : slots)
        {
            result.master_dtypes.push_back(slot.parameter.dtype);
            result.gradient_dtypes.push_back(slot.gradient.dtype);
        }

        return result;
    }
}

TEST_F(Regression_MixedPrecision, TheMasterWeightsAndTheirGradientsStayFloat32)
{
    const MixedPrecisionPolicy precision;
    const TrainingRun mixed = train(*engine, &precision);

    ASSERT_FALSE(mixed.master_dtypes.empty());
    ASSERT_GT(mixed.casts, 0u) << "nothing was narrowed, so nothing was being guarded";

    for (DType dtype : mixed.master_dtypes)
        EXPECT_EQ(dtype, DType::FLOAT32)
            << "a master weight was narrowed; fp16 updates below the weight's own "
               "spacing are discarded and the run still converges";

    for (DType dtype : mixed.gradient_dtypes)
        EXPECT_EQ(dtype, DType::FLOAT32)
            << "a gradient reached the optimizer in low precision";
}

TEST_F(Regression_MixedPrecision, TheFp32PolicyIsIndistinguishableFromNoPolicyAtAll)
{
    // The control, and the strongest form of it available: not just "zero
    // casts" but the same recorded operation count and the same losses to the
    // bit. If the pass ever starts inserting a conversion unconditionally,
    // this fails before any mixed-precision test gets the chance to look
    // correct for the wrong reason.
    const UniformPrecisionPolicy single;

    const TrainingRun none = train(*engine, nullptr);
    const TrainingRun explicit_fp32 = train(*engine, &single);

    EXPECT_EQ(none.casts, 0u);
    EXPECT_EQ(explicit_fp32.casts, 0u);
    EXPECT_EQ(none.operations, explicit_fp32.operations);

    ASSERT_EQ(none.losses.size(), explicit_fp32.losses.size());
    for (std::size_t step = 0; step < none.losses.size(); ++step)
        EXPECT_EQ(none.losses[step], explicit_fp32.losses[step]) << "step " << step;
}

TEST_F(Regression_MixedPrecision, NarrowingCostsConversionsAndNothingElseStructural)
{
    // Casts are nodes, so they show up as recorded operations - and the count
    // has to be exactly the casts, not more. More would mean a conversion
    // landed inside a fusable chain and cost the fusion as well, which is the
    // failure graph/cast_insertion.hpp's ordering argument exists to avoid and
    // which nothing else here would notice.
    const MixedPrecisionPolicy precision;

    const TrainingRun fp32 = train(*engine, nullptr);
    const TrainingRun mixed = train(*engine, &precision);

    EXPECT_EQ(mixed.operations, fp32.operations + mixed.casts)
        << "mixed precision changed the plan by more than the conversions it added";
}

TEST_F(Regression_MixedPrecision, MixedAndSinglePrecisionTrainToTheSamePlace)
{
    // The numeric claim, and the tolerance is stated rather than tuned until
    // green. fp16 carries 11 bits of significand, so one rounded value is
    // accurate to about 2^-11 = 4.9e-4 relative; a step's loss passes through
    // a handful of them, so a few times that is the honest expectation.
    //
    // The measured figure on `sa demo mlp --epochs 8 --batches 16` (a wider
    // model than this one, 128 steps) was 9.5e-4 relative, at the end of the
    // first epoch - the largest loss whose six printed decimals still carry
    // enough digits to divide. The bound below is 2e-2, about twenty times
    // that, and it is a *bound* rather than a measurement: it is chosen so the
    // test survives a change of seed or schedule, and passing it says the
    // deviation is under 2e-2 and nothing more precise than that.
    const MixedPrecisionPolicy precision;

    const TrainingRun fp32 = train(*engine, nullptr);
    const TrainingRun mixed = train(*engine, &precision);

    ASSERT_EQ(fp32.losses.size(), mixed.losses.size());
    ASSERT_GT(mixed.casts, 0u);

    // Both actually learned; comparing two runs that both diverged would agree
    // beautifully and mean nothing.
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

TEST_F(Regression_MixedPrecision, AGeluLayerIsNotFoldedUnderANarrowingPolicy)
{
    // graph/precision.hpp's rules 2 and 5 meet here, and this is the assertion
    // that they meet consistently rather than merely quietly.
    //
    // Rule 2 says a fused epilogue runs inside the GEMM's kernel in the GEMM's
    // element type, so an activation the catalog folds has to compute in the
    // GEMM's type or fusion would change the numbers. Rule 5 says GELU stays
    // wide - the tanh approximation computes x*x*x, which overflows fp16 at
    // |x| about 40.3. Since FusedForm::GEMM_BIAS_GELU exists, those two would
    // contradict each other if nothing kept them apart.
    //
    // What keeps them apart is the pass order, and it is structural rather than
    // lucky: the cast pass runs *before* fusion, so an activation the policy
    // keeps wide gets a CAST between it and the narrowed BIAS_ADD, and a chain
    // with a conversion in the middle matches no catalog entry. Under fp32 the
    // layer folds whole; under a narrowing policy it falls back to gemm+bias
    // and the GELU runs on its own, exactly as it did before the entry existed.
    //
    // If this ever starts producing a gemm+bias+gelu node, an fp16 x*x*x is
    // being computed and nothing else in the suite would say so.
    using GeluMlp = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

    GeluMlp wide_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});
    GeluMlp narrow_model(NN::Linear{FEATURES, HIDDEN}, NN::GELU{}, NN::Linear{HIDDEN, CLASSES});

    const MixedPrecisionPolicy precision;

    const Train::TrainingGraph wide = Train::build_classifier(
        wide_model, BATCH, FEATURES, CLASSES, DType::FLOAT32, Graph::FusionOptions{}, nullptr);
    const Train::TrainingGraph narrow = Train::build_classifier(
        narrow_model, BATCH, FEATURES, CLASSES, DType::FLOAT32, Graph::FusionOptions{},
        &precision);

    // The control: with nothing narrowing, the entry is reachable and taken.
    EXPECT_EQ(wide.fusion.count(Graph::FusedForm::GEMM_BIAS_GELU), 1u)
        << wide.fusion.to_string();

    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMM_BIAS_GELU), 0u)
        << "a narrowing policy folded GELU into the GEMM, which puts the tanh "
           "approximation's x*x*x into fp16 - see graph/precision.hpp rule 5\n"
        << narrow.fusion.to_string();

    // And it did not simply stop fusing: the bias still folds on both layers,
    // so the fallback is the shorter entry rather than nothing.
    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMM_BIAS), 2u)
        << narrow.fusion.to_string();
    EXPECT_GT(narrow.casts.count(), 0u) << "this arm narrowed nothing, so it proves nothing";

    // The GELU itself is still in the graph, and still in the master type -
    // which is the other half of rule 5 and the reason the fold is refused.
    for (const Graph::Node& node : narrow.graph.nodes())
        if (node.op == Ops::GELU.id)
            EXPECT_EQ(narrow.graph.value(node.outputs[0]).dtype, DType::FLOAT32);
}
