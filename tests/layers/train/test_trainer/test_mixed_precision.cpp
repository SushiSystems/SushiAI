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
 * @file test_mixed_precision.cpp
 * @brief Tests mixed precision: execution, tensor dtypes, the policy check and the float32 master
 *        weights.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
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
    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    MLP make_model(int64_t features, int64_t hidden, int64_t classes)
    {
        return MLP(NN::Linear{features, hidden}, NN::ReLU{}, NN::Linear{hidden, classes});
    }

    constexpr int64_t K_BATCH = 16;
    constexpr int64_t K_FEATURES = 8;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_GRAPH_HIDDEN = 16;
    constexpr int64_t K_BATCHES = 8;
    constexpr int64_t K_EPOCHS = 25;

    /** @brief The fraction of its first loss a learning run's last loss must stay under. */
    constexpr double K_FALL_FRACTION = 0.5;

    /** @brief The accuracy a learning run must pass. */
    constexpr double K_MIN_ACCURACY = 0.75;

    /** @brief The unit roundoff of fp16: 11 bits of significand, 2^-11. */
    constexpr double K_HALF_UNIT = 1.0 / 2048.0;

    /**
     * @brief Counts the fp16 units the loss of a step may stray from the fp32 loss.
     *
     * This is a bound over a trajectory of 200 steps and not the rounding of one pass: a handful
     * of rounded values reach each loss, and the updates they steer compound. The figure measured
     * on `sa demo mlp --epochs 8 --batches 16` was 9.5e-4 relative, 2 units.
     */
    constexpr double K_MIXED_LOSS_UNITS = 40.0;

    /** @brief Bounds the relative deviation of a mixed loss from the single-precision one. */
    constexpr double K_MIXED_LOSS_BOUND = K_MIXED_LOSS_UNITS * K_HALF_UNIT;

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

    /** @brief Builds, binds, initialises and compiles the mixed-precision classifier. */
    class Medium_MixedPrecision : public SaTest::EngineTest
    {
        protected:
            /** @brief Prepares the graph, the state and the pool under the mixed policy. */
            void prepare()
            {
                SA_ASSERT_OK_AND_ASSIGN(
                    built, Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES,
                                                   DType::Float32, Graph::FusionOptions{},
                                                   &precision));
                SA_ASSERT_OK_AND_ASSIGN(state, NN::ModelState::create(*engine, model));
                pool = std::make_unique<TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));
                SA_EXPECT_OK(NN::initialize(backend(), *state));
                SA_ASSERT_OK_AND_ASSIGN(
                    optimizer, Optim::AdamW::make(backend(), slots, optimizer_options()));
            }

            /** @brief Makes the trainer under the mixed policy over @p epochs epochs. */
            void make_trainer(int64_t epochs)
            {
                Train::TrainerOptions options;
                options.epochs = epochs;
                options.precision = &precision;
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::classifier_objective(built));
                SA_ASSERT_OK_AND_ASSIGN(
                    trainer, Train::Trainer::make(backend(), built, *pool, *objective,
                                                  *optimizer, options));
            }

            /** @brief Makes the dataset of eight batches. */
            void make_dataset()
            {
                SA_ASSERT_OK_AND_ASSIGN(
                    dataset, Data::SyntheticClassification::make(backend(), data_options()));
            }

            MLP model = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);   /**< @brief The model. */
            MixedPrecisionPolicy precision;                            /**< @brief The policy. */
            Train::TrainingGraph built;                                /**< @brief The graph. */
            std::optional<NN::ModelState> state;                       /**< @brief The parameters. */
            std::unique_ptr<TensorPool> pool;                          /**< @brief The values. */
            std::vector<Optim::ParameterSlot> slots;                   /**< @brief The slots. */
            std::unique_ptr<Optim::AdamW> optimizer;                   /**< @brief The optimizer. */
            std::unique_ptr<Train::CrossEntropyObjective> objective;   /**< @brief The objective. */
            std::unique_ptr<Train::Trainer> trainer;                   /**< @brief The trainer. */
            std::unique_ptr<Data::SyntheticClassification> dataset;    /**< @brief The data. */
    };
}

/** @brief Tests that a model with fp16 products ends with a loss under half its first. */
TEST_F(Medium_MixedPrecision, AModelWithFP16ProductsStillLearns)
{
    // Correctness is asserted; speed is not; see tests/layers/train/README.md.
    prepare();
    ASSERT_GT(built.casts.count(), 0u) << "the mixed policy inserted nothing";
    make_trainer(25);
    make_dataset();

    SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));

    EXPECT_EQ(report.compile_count, 1u);
    EXPECT_TRUE(std::isfinite(report.first_loss));
    EXPECT_TRUE(std::isfinite(report.last_loss));
    EXPECT_LT(report.last_loss, K_FALL_FRACTION * report.first_loss);
    ASSERT_NE(report.last_metrics.find("accuracy"), nullptr);
    EXPECT_GT(report.last_metrics.find("accuracy")->value, K_MIN_ACCURACY);
}

/** @brief Tests that every half cast destination is materialised as a half tensor. */
TEST_F(Medium_MixedPrecision, TheCopiesAreTwoByteBuffers)
{
    // Tensors must carry the dtypes the IR claims; see tests/layers/train/README.md.
    prepare();

    // The trainer's compile materialises the destinations, so a pool bound alone holds none.
    make_trainer(1);

    std::vector<InsertedCast> halves;
    const auto& casts = built.casts.casts();
    std::copy_if(casts.begin(), casts.end(), std::back_inserter(halves),
                 [](const InsertedCast& cast) { return cast.to == DType::Half; });

    // At least one genuine half buffer exists, so the fp16 path is real.
    ASSERT_FALSE(halves.empty());
    EXPECT_TRUE(std::all_of(halves.begin(), halves.end(),
                            [&](const InsertedCast& cast)
                            { return pool->is_bound(cast.destination); }))
        << "a cast destination was never materialised";
    EXPECT_TRUE(std::all_of(halves.begin(), halves.end(),
                            [&](const InsertedCast& cast)
                            { return pool->tensor(cast.destination).dtype == DType::Half; }));
}

/** @brief Tests that one step over the half copies gives a finite loss. */
TEST_F(Medium_MixedPrecision, AStepOverTheHalfCopiesIsFinite)
{
    prepare();
    make_trainer(1);
    make_dataset();

    // SushiBLAS refuses a mixed-dtype gemm by name; see tests/layers/train/README.md.
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset->read(0, dataset_batch)));
    SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics, trainer->run_step(dataset_batch));
    EXPECT_TRUE(std::isfinite(metrics.loss));
}

TEST_F(Medium_MixedPrecision, TheTrainerRefusesAGraphItsPolicyDisowns)
{
    // The graph skipped the cast pass on purpose; see tests/layers/train/README.md.
    MLP unnarrowed = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph wide,
                            Train::build_classifier(unnarrowed, K_BATCH, K_FEATURES, K_CLASSES));

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState wide_state,
                            NN::ModelState::create(*engine, unnarrowed));
    TensorPool wide_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> wide_slots,
                            Train::bind_parameters(backend(), wide, wide_state, wide_pool));
    SA_EXPECT_OK(NN::initialize(backend(), wide_state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> wide_optimizer,
                            Optim::AdamW::make(backend(), wide_slots, optimizer_options()));

    Train::TrainerOptions options;
    options.precision = &precision;

    const auto wide_objective = SaTest::classifier_objective(wide);
    SA_EXPECT_FAILURE(Train::Trainer::make(backend(), wide, wide_pool, wide_objective,
                                           *wide_optimizer, options),
                      StatusCode::InvalidDescription, "does not agree with the precision policy");

    // The same graph with the policy it was actually built under is accepted.
    const UniformPrecisionPolicy single;
    Train::TrainerOptions consistent;
    consistent.precision = &single;
    SA_EXPECT_OK(Train::Trainer::make(backend(), wide, wide_pool, wide_objective,
                                      *wide_optimizer, consistent));
}

TEST_F(Medium_MixedPrecision, AccuracyRefusesLogitsItCannotRead)
{
    // A half buffer must not be read as float; see tests/layers/train/README.md.
    MLP unnarrowed = make_model(K_FEATURES, K_HIDDEN, K_CLASSES);
    SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph narrowed,
                            Train::build_classifier(unnarrowed, K_BATCH, K_FEATURES, K_CLASSES));

    // Deliberately narrow the logits: no observed list, so nothing pins them.
    Graph::CastOptions casts;
    casts.policy = &precision;
    SA_EXPECT_OK(Graph::insert_casts(narrowed.graph, casts));

    ASSERT_EQ(narrowed.graph.value(narrowed.spec.logits).dtype, DType::Half);

    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState narrow_state,
                            NN::ModelState::create(*engine, unnarrowed));
    TensorPool narrow_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> narrow_slots,
                            Train::bind_parameters(backend(), narrowed, narrow_state,
                                                   narrow_pool));
    SA_EXPECT_OK(NN::initialize(backend(), narrow_state));

    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Optim::AdamW> narrow_optimizer,
                            Optim::AdamW::make(backend(), narrow_slots, optimizer_options()));

    Train::TrainerOptions options;
    options.track_metrics = true;
    const auto narrow_objective = SaTest::classifier_objective(narrowed);
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Train::Trainer> narrow_trainer,
                            Train::Trainer::make(backend(), narrowed, narrow_pool,
                                                 narrow_objective, *narrow_optimizer, options));

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> narrow_data,
                            Data::SyntheticClassification::make(backend(), data_options()));
    Data::Batch dataset_batch;
    ASSERT_TRUE(SaTest::status_is_ok(narrow_data->read(0, dataset_batch)));
    SA_EXPECT_FAILURE(narrow_trainer->run_step(dataset_batch), StatusCode::InvalidDescription,
                      "needs a host read of the logits");
}

namespace
{
    /** @brief Names which dtype list of a mixed run a row reads. */
    enum class DtypeList
    {
        Masters,
        Gradients
    };

    class Medium_MixedPrecisionRegressions : public SaTest::EngineTest {};

    /** @brief Reads each master weight and gradient dtype of a mixed run. */
    class Medium_MixedPrecisionDtypes : public ::testing::TestWithParam<DtypeList> {};

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
    [[nodiscard]] Result<TrainingRun> train(SushiAI::ExecutionBackend& backend,
                                            const Graph::PrecisionPolicy* precision)
    {
        MLP model(NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::ReLU{}, NN::Linear{K_GRAPH_HIDDEN, K_CLASSES});

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES, DType::Float32,
                                              Graph::FusionOptions{}, precision));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), model));
        TensorPool pool(backend.engine());
        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, built, state, pool));
        SA_TRY(NN::initialize(backend, state));

        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> optimizer,
                      Optim::AdamW::make(backend, slots, optimizer_options()));

        Train::TrainerOptions options;
        options.precision = precision;
        const auto objective = SaTest::classifier_objective(built);
        SA_TRY_ASSIGN(std::unique_ptr<Train::Trainer> trainer,
                      Train::Trainer::make(backend, built, pool, objective, *optimizer,
                                           options));

        Data::SyntheticOptions data;
        data.batch_size = K_BATCH;
        data.batches = K_BATCHES;
        data.features = K_FEATURES;
        data.classes = K_CLASSES;
        SA_TRY_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                      Data::SyntheticClassification::make(backend, data));

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

    /** @brief Returns the largest relative distance of @p mixed from @p single, step by step. */
    double worst_relative_deviation(const std::vector<double>& single,
                                    const std::vector<double>& mixed)
    {
        double worst = 0.0;
        for (std::size_t step = 0; step < single.size(); ++step)
            worst = std::max(worst, std::abs(mixed[step] - single[step]) / std::abs(single[step]));

        return worst;
    }

    /** @brief Names a row of the dtype table. */
    std::string dtype_list_name(const ::testing::TestParamInfo<DtypeList>& info)
    {
        return info.param == DtypeList::Masters ? "Masters" : "Gradients";
    }

    /** @brief Builds the GELU classifier with no policy and with the mixed policy. */
    class Small_MixedPrecisionGelu : public ::testing::Test
    {
        protected:
            /** @brief Traces the two graphs. */
            void SetUp() override
            {
                SA_ASSERT_OK_AND_ASSIGN(
                    wide, Train::build_classifier(wide_model, K_BATCH, K_FEATURES, K_CLASSES,
                                                  DType::Float32, Graph::FusionOptions{},
                                                  nullptr));
                SA_ASSERT_OK_AND_ASSIGN(
                    narrow, Train::build_classifier(narrow_model, K_BATCH, K_FEATURES, K_CLASSES,
                                                    DType::Float32, Graph::FusionOptions{},
                                                    &precision));
            }

            using GELUMLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;

            GELUMLP wide_model{NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::GELU{},
                               NN::Linear{K_GRAPH_HIDDEN, K_CLASSES}};      /**< @brief Unnarrowed. */
            GELUMLP narrow_model{NN::Linear{K_FEATURES, K_GRAPH_HIDDEN}, NN::GELU{},
                                 NN::Linear{K_GRAPH_HIDDEN, K_CLASSES}};    /**< @brief Narrowed. */
            MixedPrecisionPolicy precision;                                 /**< @brief The policy. */
            Train::TrainingGraph wide;                                      /**< @brief No policy. */
            Train::TrainingGraph narrow;                                    /**< @brief Mixed policy. */
    };
}

/**
 * @brief Tests that every master weight or gradient is float32 after a mixed-precision run.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_MixedPrecisionDtypes, StayFloat32)
{
    const MixedPrecisionPolicy precision;
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed,
                            train(SaTest::TestEnvironment::backend(), &precision));

    ASSERT_FALSE(mixed.master_dtypes.empty());
    ASSERT_GT(mixed.casts, 0u) << "nothing was narrowed, so nothing was being guarded";

    const std::vector<DType>& dtypes = GetParam() == DtypeList::Masters
        ? mixed.master_dtypes : mixed.gradient_dtypes;
    EXPECT_TRUE(std::all_of(dtypes.begin(), dtypes.end(),
                            [](DType dtype) { return dtype == DType::Float32; }))
        << "a master weight was narrowed, or a gradient reached the optimizer in low precision; "
           "fp16 updates below the weight's own spacing are discarded and the run still converges";
}

INSTANTIATE_TEST_SUITE_P(Lists, Medium_MixedPrecisionDtypes,
                         ::testing::Values(DtypeList::Masters, DtypeList::Gradients),
                         dtype_list_name);

/**
 * @brief Tests that the fp32 policy adds no cast and matches no policy bit for bit.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, TheFP32PolicyIsIndistinguishableFromNoPolicyAtAll)
{
    const UniformPrecisionPolicy single;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun none, train(backend(), nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun explicit_fp32, train(backend(), &single));

    EXPECT_EQ(none.casts, 0u);
    EXPECT_EQ(explicit_fp32.casts, 0u);
    EXPECT_EQ(none.operations, explicit_fp32.operations);
    EXPECT_EQ(none.losses, explicit_fp32.losses);
}

/**
 * @brief Tests that mixed precision adds exactly as many operations as casts.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, NarrowingCostsConversionsAndNothingElseStructural)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(backend(), nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(backend(), &precision));

    EXPECT_EQ(mixed.operations, fp32.operations + mixed.casts)
        << "mixed precision changed the plan by more than the conversions it added";
}

/**
 * @brief Tests that mixed and single precision losses stay within K_MIXED_LOSS_BOUND at every step.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_MixedPrecisionRegressions, MixedAndSinglePrecisionTrainToTheSamePlace)
{
    const MixedPrecisionPolicy precision;

    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun fp32, train(backend(), nullptr));
    SA_ASSERT_OK_AND_ASSIGN(const TrainingRun mixed, train(backend(), &precision));

    ASSERT_EQ(fp32.losses.size(), mixed.losses.size());
    ASSERT_GT(mixed.casts, 0u);

    // Both runs learned before they are compared; see tests/layers/train/README.md.
    EXPECT_LT(fp32.losses.back(), K_FALL_FRACTION * fp32.losses.front());
    EXPECT_LT(mixed.losses.back(), K_FALL_FRACTION * mixed.losses.front());
    EXPECT_GT(mixed.accuracy, K_MIN_ACCURACY);

    EXPECT_TRUE(std::all_of(mixed.losses.begin(), mixed.losses.end(),
                            [](double loss) { return std::isfinite(loss); }));
    EXPECT_LT(worst_relative_deviation(fp32.losses, mixed.losses), K_MIXED_LOSS_BOUND);
}

/** @brief Tests that a narrowing policy does not fold GELU into the GEMM, and that none does. */
TEST_F(Small_MixedPrecisionGelu, AGELULayerIsNotFoldedUnderANarrowingPolicy)
{
    // The control: with nothing narrowing, the entry is reachable and taken.
    EXPECT_EQ(wide.fusion.count(Graph::FusedForm::GEMMBiasGELU), 1u) << wide.fusion.to_string();

    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMMBiasGELU), 0u)
        << "a narrowing policy folded GELU into the GEMM, which puts the tanh "
           "approximation's x*x*x into fp16 - see graph/precision.hpp rule 5\n"
        << narrow.fusion.to_string();
}

/** @brief Tests that the bias still folds into both GEMMs under the narrowing policy. */
TEST_F(Small_MixedPrecisionGelu, TheBiasStillFoldsOnBothLayersUnderANarrowingPolicy)
{
    // The fallback is the shorter catalog entry and not an unfused graph.
    EXPECT_GT(narrow.casts.count(), 0u) << "this arm narrowed nothing, so it proves nothing";
    EXPECT_EQ(narrow.fusion.count(Graph::FusedForm::GEMMBias), 2u) << narrow.fusion.to_string();
}

/** @brief Tests that the GELU stays in the graph, in the master type. */
TEST_F(Small_MixedPrecisionGelu, TheGELUOutputStaysFloat32UnderANarrowingPolicy)
{
    const auto& nodes = narrow.graph.nodes();
    const auto gelu = [](const Graph::Node& node) { return node.op == Ops::K_GELU.id; };

    ASSERT_NE(std::find_if(nodes.begin(), nodes.end(), gelu), nodes.end())
        << "the GELU left the graph";

    // The other half of rule 5 and the reason the fold is refused.
    EXPECT_TRUE(std::all_of(nodes.begin(), nodes.end(),
                            [&](const Graph::Node& node)
                            { return !gelu(node) ||
                                     narrow.graph.value(node.outputs[0]).dtype == DType::Float32; }));
}
