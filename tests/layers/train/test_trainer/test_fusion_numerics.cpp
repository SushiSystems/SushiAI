/**************************************************************************/
/* test_fusion_numerics.cpp                                               */
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
 * @file test_fusion_numerics.cpp
 * @brief Tests that the fused graph trains byte for byte like the unfused graph it replaces.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/elementwise/elementwise.hpp>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/engine/execution_mode.hpp>
#include <SushiBLAS/engine/runtime_scratch_source.hpp>
#include <SushiBLAS/engine/scratch_pool.hpp>
#include <SushiBLAS/engine/late_count_hold.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiBLAS/launch/device_policy.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/api/graph/graph.hpp>
#include <SushiRuntime/api/graph/plan_view.hpp>
#include <SushiRuntime/core/result.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiRuntime/sushiruntime.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/seeded_noise/host_noise.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/tolerance.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;
using Graph::FusedForm;
using Graph::FusionOptions;
using Graph::TensorPool;

namespace
{
    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_HIDDEN = 12;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;
    constexpr int64_t K_EPOCHS = 3;

    /** @brief Holds the rows of the spill probe. */
    constexpr int64_t K_ROWS = 4;

    /** @brief Holds the inputs of a row of the spill probe. */
    constexpr int64_t K_IN_DIM = 3;

    /** @brief Holds the outputs of a row of the spill probe. */
    constexpr int64_t K_OUT_DIM = 5;

    /**
     * @brief Holds the pre-activation below which gelu(x) and x are told apart. At -0.5 they differ
     *        by 0.35, so any element under the cut separates them by more than K_SEPARATION.
     */
    constexpr double K_NEGATIVE_CUT = -0.5;

    /** @brief Holds the distance two tensors must exceed to count as different at the cut. */
    constexpr double K_SEPARATION = 0.1;

    /** @brief The activations a row of the training comparisons builds its model with. */
    enum class Activation
    {
        ReLU,
        Tanh,
        Sigmoid,
        GELU
    };

    /** @brief The synthetic problem every comparison below trains on. */
    Data::SyntheticOptions data_options()
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        options.seed = 0x0F05'ED00'1234ull;
        return options;
    }

    /**
     * @struct TrainedRun
     * @brief One whole training run, reduced to what a comparison needs.
     */
    struct TrainedRun
    {
        std::vector<std::vector<unsigned char>> parameters;
        std::vector<std::string> names;
        std::vector<double> losses;
        std::vector<double> accuracies;
        std::size_t operations_per_step = 0;
        std::size_t fused_subtrees = 0;
        std::size_t selector_refusals = 0;
    };

    /**
     * @brief Trains one model end to end and writes everything comparable into @p run.
     * @param run Left partly filled when a step of the run fails, which fails the test.
     * @param selector Null selects this machine's own device.
     * @see tests/layers/train/README.md
     */
    template<typename M>
    void train(M& model, FusionOptions fusion, TrainedRun& run,
               const Graph::FusionSelector* selector = nullptr)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

        SA_ASSERT_OK_AND_ASSIGN(Train::TrainingGraph built,
                                Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES,
                                                        DType::Float32, fusion));

        SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, model));
        TensorPool pool(engine);
        SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                                Train::bind_parameters(backend, built, state, pool));
        ASSERT_TRUE(SaTest::status_is_ok(NN::initialize(backend, state)));

        Optim::AdamWOptions adam;
        adam.learning_rate = 0.02;
        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::AdamW> optimizer,
                                Optim::AdamW::make(backend, slots, adam));

        Train::TrainerOptions options;
        options.epochs = K_EPOCHS;
        options.fusion_selector = selector;
        const auto objective = SaTest::classifier_objective(built);
        SA_ASSERT_OK_AND_ASSIGN(
            const std::unique_ptr<Train::Trainer> trainer,
            Train::Trainer::make(backend, built, pool, objective, *optimizer, options));

        SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                                Data::SyntheticClassification::make(backend, data_options()));

        run.fused_subtrees = built.fusion.fused().size();

        for (const Graph::FusionSelection& selection : trainer->step().selections())
            if (selection.chosen != selection.matched) ++run.selector_refusals;

        SA_ASSERT_OK_AND_ASSIGN(const Train::RunReport report, trainer->fit(*dataset));
        run.operations_per_step = report.operations_per_step;

        for (int64_t epoch = 0; epoch < K_EPOCHS; ++epoch)
            for (std::size_t batch = 0; batch < dataset->batch_count(); ++batch)
            {
                Data::Batch dataset_batch;
                ASSERT_TRUE(SaTest::status_is_ok(dataset->read(batch, dataset_batch)));
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                        trainer->run_step(dataset_batch));
                run.losses.push_back(metrics.loss);
                run.accuracies.push_back(metrics.metrics.value_or("accuracy", 0.0));
            }

        for (const Optim::ParameterSlot& slot : slots)
        {
            run.names.push_back(slot.name);
            run.parameters.push_back(SaTest::read_tensor_bytes(slot.parameter));
        }
    }

    /** @brief Trains the two-layer model with @p Act between its layers, see train(). */
    template<typename Act>
    void train_stack(FusionOptions fusion, TrainedRun& run, const Graph::FusionSelector* selector)
    {
        using MLP = NN::Sequential<NN::Linear, Act, NN::Linear>;

        MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, Act{}, NN::Linear{K_HIDDEN, K_CLASSES});
        train(model, fusion, run, selector);
    }

    /** @brief Trains the two-layer model of @p activation, see train(). */
    void train_with(Activation activation, FusionOptions fusion, TrainedRun& run,
                    const Graph::FusionSelector* selector = nullptr)
    {
        switch (activation)
        {
            case Activation::ReLU:
                train_stack<NN::ReLU>(fusion, run, selector);
                break;
            case Activation::Tanh:
                train_stack<NN::Tanh>(fusion, run, selector);
                break;
            case Activation::Sigmoid:
                train_stack<NN::Sigmoid>(fusion, run, selector);
                break;
            case Activation::GELU:
                train_stack<NN::GELU>(fusion, run, selector);
                break;
        }
    }

    /** @brief Asserts two runs agree byte for byte, naming where they do not. */
    void expect_runs_agree(const TrainedRun& a, const TrainedRun& b, const std::string& what)
    {
        ASSERT_EQ(a.names, b.names);
        ASSERT_EQ(a.losses.size(), b.losses.size());
        ASSERT_FALSE(a.losses.empty());

        for (std::size_t step = 0; step < a.losses.size(); ++step)
            EXPECT_EQ(a.losses[step], b.losses[step])
                << what << ": the loss diverged at step " << step << " (" << a.losses[step]
                << " vs " << b.losses[step] << "). Find the expression that moved; do not "
                << "widen this comparison.";

        for (std::size_t step = 0; step < a.accuracies.size(); ++step)
            EXPECT_EQ(a.accuracies[step], b.accuracies[step])
                << what << ": accuracy diverged at step " << step;

        for (std::size_t i = 0; i < a.parameters.size(); ++i)
            EXPECT_EQ(SaTest::first_difference(a.parameters[i], b.parameters[i]),
                      a.parameters[i].size())
                << what << ": parameter '" << a.names[i] << "' differs";
    }

    /** @brief The unfused control. */
    FusionOptions off()
    {
        FusionOptions options;
        options.enabled = false;
        return options;
    }

    /** @brief A selector that declines every matched form. */
    class RefuseEverything final : public Graph::FusionSelector
    {
        public:
            /** @brief Answers no form, with a reason. */
            Graph::FusionChoice choose(const Graph::FusionQuery&) const override
            {
                return Graph::FusionChoice{FusedForm::None, "this device wants none of it"};
            }
    };

    /**
     * @brief Returns a device profile that is not this machine's.
     * @see tests/layers/train/README.md
     */
    SushiBLAS::Device::Profile pretend_device(bool fp64)
    {
        SushiBLAS::Device::Profile profile;
        profile.is_cpu = false;
        profile.has_dedicated_local_memory = true;
        profile.supports_fp64 = fp64;
        profile.local_mem_bytes = 48 * 1024;
        profile.max_work_group_size = 1024;
        profile.sub_group_size = 32;
        profile.compute_units = 10;
        return profile;
    }

    /** @brief Holds what the spill probe computed on the device and on the host. */
    struct SpillRun
    {
        std::vector<double> expected;    /**< @brief Holds b + x w, worked on the host. */
        std::vector<double> magnitude;   /**< @brief Holds |b| + the sum of |x w| per element. */
        std::vector<double> spilled;     /**< @brief Holds the spill buffer after the run. */
        std::vector<double> activated;   /**< @brief Holds the activation after the run. */
        std::size_t emitted = 0;         /**< @brief Holds the operations the lowering emitted. */
    };

    /** @brief Works b + x w for the probe's inputs on the host, with the magnitude of each sum. */
    void work_pre_activation(const std::vector<double>& x, const std::vector<double>& w,
                             const std::vector<double>& b, SpillRun& run)
    {
        for (int64_t row = 0; row < K_ROWS; ++row)
            for (int64_t col = 0; col < K_OUT_DIM; ++col)
            {
                double value = b[static_cast<std::size_t>(col)];
                double magnitude = std::abs(value);
                for (int64_t k = 0; k < K_IN_DIM; ++k)
                {
                    const double product = x[static_cast<std::size_t>(row * K_IN_DIM + k)]
                                         * w[static_cast<std::size_t>(col * K_IN_DIM + k)];
                    value += product;
                    magnitude += std::abs(product);
                }

                run.expected.push_back(value);
                run.magnitude.push_back(magnitude);
            }
    }

    /**
     * @brief Lowers a fused bias-add and GELU with @p selector and reads the spill.
     * @param selector Null selects this machine's own device.
     */
    void run_spill(const Graph::FusionSelector* selector, SpillRun& run)
    {
        SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
        SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

        Graph::Graph graph;
        const Graph::GraphBuilder builder(graph);

        const Graph::ValueID x = builder.input(Shape{K_ROWS, K_IN_DIM}, "x");
        const Graph::ValueID w = builder.parameter(Shape{K_OUT_DIM, K_IN_DIM}, "w");
        const Graph::ValueID b = builder.parameter(Shape{K_OUT_DIM}, "b");
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID pre,
                                builder.bias_add(builder.matmul(x, w, false, true), b));
        SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID out, builder.gelu(pre));

        // `pre` is unobserved; the fold rests on the spill; see tests/layers/train/README.md.
        const std::vector<Graph::ValueID> observed{out};
        FusionOptions options;
        options.observed =
            SushiRuntime::span<const Graph::ValueID>(observed.data(), observed.size());

        const Graph::FusionReport report = Graph::fuse(graph, options);
        ASSERT_EQ(report.count(FusedForm::GEMMBiasGELU), 1u) << report.to_string();

        TensorPool pool(engine);
        ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

        SaTest::HostNoise noise(0xC0FFEEu);
        std::vector<double> x_values(static_cast<std::size_t>(K_ROWS * K_IN_DIM));
        std::vector<double> w_values(static_cast<std::size_t>(K_OUT_DIM * K_IN_DIM));
        std::vector<double> b_values(static_cast<std::size_t>(K_OUT_DIM));
        noise.fill(SushiRuntime::span<double>(x_values.data(), x_values.size()), 2.0);
        noise.fill(SushiRuntime::span<double>(w_values.data(), w_values.size()), 2.0);
        noise.fill(SushiRuntime::span<double>(b_values.data(), b_values.size()), 2.0);

        SaTest::write_tensor(pool.tensor(x), x_values);
        SaTest::write_tensor(pool.tensor(w), w_values);
        SaTest::write_tensor(pool.tensor(b), b_values);

        const Graph::DeviceFusionSelector own = Graph::DeviceFusionSelector::for_backend(backend);
        SaTest::TestGraph recording(backend);
        Graph::Lowering lowering(recording.recorder(), engine.rng(), pool,
                                 selector == nullptr ? own : *selector);
        ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
        run.emitted = lowering.emitted_operations();

        // A declined epilogue adds the bias through a ones vector, whose fill the pool queued.
        SA_EXPECT_OK(pool.flush_constants(backend));
        SA_EXPECT_OK(recording.run());
        SaTest::TestEnvironment::runtime().wait().value();

        run.spilled = SaTest::read_tensor(pool.tensor(pre));
        run.activated = SaTest::read_tensor(pool.tensor(out));
        work_pre_activation(x_values, w_values, b_values, run);
    }

    /** @brief Returns how many elements of @p run's spill lie outside the bound of their sum. */
    std::size_t spill_errors(const SpillRun& run)
    {
        std::size_t outside = 0;
        for (std::size_t index = 0; index < run.expected.size(); ++index)
            outside += TestSupport::measure_sum_error<float>(
                           run.spilled[index], run.expected[index], run.magnitude[index],
                           static_cast<std::size_t>(K_IN_DIM) + 1).is_within() ? 0u : 1u;

        return outside;
    }

    /** @brief Returns how many elements of @p run's pre-activation lie under K_NEGATIVE_CUT. */
    std::size_t negative_count(const SpillRun& run)
    {
        return static_cast<std::size_t>(std::count_if(
            run.expected.begin(), run.expected.end(),
            [](double value) { return value < K_NEGATIVE_CUT; }));
    }

    /** @brief Returns how many elements under K_NEGATIVE_CUT hold a spill apart from the activation. */
    std::size_t separated_count(const SpillRun& run)
    {
        std::size_t separated = 0;
        for (std::size_t index = 0; index < run.expected.size(); ++index)
            separated += run.expected[index] < K_NEGATIVE_CUT &&
                std::abs(run.spilled[index] - run.activated[index]) > K_SEPARATION ? 1u : 0u;

        return separated;
    }

    /** @brief Expects the spill to hold the pre-activation and to differ from the activation. */
    void expect_spill_is_the_pre_activation(const SpillRun& run)
    {
        ASSERT_EQ(run.spilled.size(), static_cast<std::size_t>(K_ROWS * K_OUT_DIM));

        EXPECT_EQ(spill_errors(run), 0u) << "the spill is not the pre-activation";

        // Only a negative element separates them; see tests/layers/train/README.md.
        EXPECT_GT(negative_count(run), 0u)
            << "no element was negative enough for gelu(x) and x to differ";
        EXPECT_EQ(separated_count(run), negative_count(run))
            << "a negative element's spill holds the activated value";
    }

    /** @brief One activation row of the training comparisons. */
    struct ActivationRow
    {
        Activation activation;   /**< @brief Holds the activation of the model. */
        const char* name;        /**< @brief Names the row in the test name. */
    };

    /** @brief One row of the disabled-form comparison. */
    struct DisabledRow
    {
        Activation activation;   /**< @brief Holds the activation of the model. */
        FusedForm form;          /**< @brief Holds the catalog form the run switches off. */
        const char* name;        /**< @brief Names the row in the test name. */
    };

    /** @brief Names a row of the activation table. */
    std::string activation_name(const ::testing::TestParamInfo<ActivationRow>& info)
    {
        return info.param.name;
    }

    /** @brief Names a row of the disabled-form table. */
    std::string disabled_name(const ::testing::TestParamInfo<DisabledRow>& info)
    {
        return info.param.name;
    }

    /** @brief Trains each activation fused and unfused and compares them. */
    class Medium_FusionTraining : public ::testing::TestWithParam<ActivationRow> {};

    /** @brief Trains each activation with and without a selector that refuses every form. */
    class Medium_FusionRefusal : public ::testing::TestWithParam<ActivationRow> {};

    /** @brief Trains each activation with and without its catalog form. */
    class Medium_FusionDisabled : public ::testing::TestWithParam<DisabledRow> {};

    /** @brief Provides the cases that train, record or read a spill on an engine of their own. */
    class Medium_Fusion : public ::testing::Test {};

    /** @brief Provides the fusion report of a graph, read without a device. */
    class Small_FusionReport : public ::testing::Test {};

    /** @brief Provides the decisions of the selector over a described device. */
    class Small_FusionSelector : public ::testing::Test {};
}

/**
 * @brief Tests that a model trains to identical bytes with fusion on and off, for each activation.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_FusionTraining, TrainsIdenticallyFusedAndUnfused)
{
    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train_with(GetParam().activation, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train_with(GetParam().activation, off(), plain));

    ASSERT_GT(fused.fused_subtrees, 0u);
    ASSERT_EQ(plain.fused_subtrees, 0u);

    expect_runs_agree(fused, plain, "fusing the forward and backward graph");
}

INSTANTIATE_TEST_SUITE_P(Activations, Medium_FusionTraining,
                         ::testing::Values(ActivationRow{Activation::ReLU, "ReLU"},
                                           ActivationRow{Activation::Tanh, "Tanh"},
                                           ActivationRow{Activation::Sigmoid, "Sigmoid"},
                                           ActivationRow{Activation::GELU, "GELU"}),
                         activation_name);

/** @brief Tests that the GELU model trains, so agreeing with the control proves something. */
TEST_F(Medium_Fusion, TheGELUModelLowersItsLoss)
{
    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train_with(Activation::GELU, FusionOptions{}, fused));

    ASSERT_GE(fused.losses.size(), 2u);
    EXPECT_LT(fused.losses.back(), fused.losses.front());
}

/** @brief Tests that the GELU model fuses its two layers and the gradient tail. */
TEST_F(Small_FusionReport, TheGELUModelFusesTwoLayersAndTheGradientTail)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    // gemm+bias+gelu, gemm+bias and the cross-entropy gradient tail.
    EXPECT_EQ(built.fusion.fused().size(), 3u) << built.fusion.to_string();
}

/** @brief Tests that the GELU fold is reported as one fold, with no refusal for its gradient. */
TEST_F(Small_FusionReport, TheGELUFoldIsReportedAsAFoldAndNotAsARefusal)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    EXPECT_EQ(built.fusion.count(FusedForm::GEMMBiasGELU), 1u) << built.fusion.to_string();

    const auto& declined = built.fusion.declined();
    EXPECT_TRUE(std::none_of(declined.begin(), declined.end(),
                             [](const Graph::DeclinedFusion& decline)
                             { return decline.reason ==
                                      Graph::FusionDecline::GradientNeedsPreActivation; }))
        << built.fusion.to_string();
}

/** @brief Tests that gelu_backward reads the value the fused GELU spills. */
TEST_F(Small_FusionReport, TheGELUBackwardReadsTheSpilledValue)
{
    using MLP = NN::Sequential<NN::Linear, NN::GELU, NN::Linear>;
    MLP model(NN::Linear{K_FEATURES, K_HIDDEN}, NN::GELU{}, NN::Linear{K_HIDDEN, K_CLASSES});

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_classifier(model, K_BATCH, K_FEATURES, K_CLASSES));

    // gelu_backward must read the spilled value; see tests/layers/train/README.md.
    const auto& fused = built.fusion.fused();
    const auto gelu = std::find_if(fused.begin(), fused.end(),
                                   [](const Graph::FusedSubtree& subtree)
                                   { return subtree.form == FusedForm::GEMMBiasGELU; });
    ASSERT_NE(gelu, fused.end());
    ASSERT_NE(gelu->spilled, Graph::K_NO_VALUE);

    const Graph::NodeID producer = built.graph.value(gelu->spilled).producer;
    ASSERT_NE(producer, Graph::K_NO_NODE) << "nothing writes the spill";
    EXPECT_EQ(built.graph.node(producer).op, Ops::K_GEMM_BIAS_GELU.id);

    const auto& nodes = built.graph.nodes();
    EXPECT_TRUE(std::any_of(nodes.begin(), nodes.end(),
                            [&](const Graph::Node& node)
                            { return node.op == Ops::K_GELU_BACKWARD.id &&
                                     node.inputs.at(1) == gelu->spilled; }))
        << "gelu_backward does not read the spilled pre-activation, so the spill is "
           "buying nothing and the fold is the silently wrong one";
}

/**
 * @brief Tests that the spill buffer holds the host-computed pre-activation.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_Fusion, TheSpillHoldsThePreActivationItself)
{
    SpillRun run;
    ASSERT_NO_FATAL_FAILURE(run_spill(nullptr, run));

    ASSERT_EQ(1u, run.emitted)
        << "this machine's device declined the fused form, so the epilogue never ran and "
           "this test is not exercising what it claims to";
    expect_spill_is_the_pre_activation(run);
}

/** @brief Tests that a refused GELU epilogue still fills the spill with the pre-activation. */
TEST_F(Medium_Fusion, ADeclinedEpilogueStillFillsTheSpillWithThePreActivation)
{
    const RefuseEverything refuse;
    SpillRun run;
    ASSERT_NO_FATAL_FAILURE(run_spill(&refuse, run));

    expect_spill_is_the_pre_activation(run);
}

/**
 * @brief Tests that a selector refusing every form trains to the bytes of the taken form.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_FusionRefusal, ASelectorMayRefuseAMatchedFormAndTheNumbersDoNotMove)
{
    const RefuseEverything refuse;
    TrainedRun taken;
    ASSERT_NO_FATAL_FAILURE(train_with(GetParam().activation, FusionOptions{}, taken));
    TrainedRun refused;
    ASSERT_NO_FATAL_FAILURE(
        train_with(GetParam().activation, FusionOptions{}, refused, &refuse));

    // The graph fused identically; only the spelling differs.
    EXPECT_EQ(taken.fused_subtrees, refused.fused_subtrees);
    EXPECT_EQ(refused.selector_refusals, refused.fused_subtrees);
    EXPECT_EQ(taken.selector_refusals, 0u)
        << "this machine's own device declined a form; the measurement notes assume it did not";

    EXPECT_GT(refused.operations_per_step, taken.operations_per_step);

    expect_runs_agree(taken, refused, "a selector that refuses every form");
}

INSTANTIATE_TEST_SUITE_P(Forms, Medium_FusionRefusal,
                         ::testing::Values(ActivationRow{Activation::ReLU, "ReLU"},
                                           ActivationRow{Activation::GELU, "GELU"}),
                         activation_name);

/**
 * @brief Tests that switching one catalog form off falls back to the rest and moves no byte.
 * @see tests/layers/train/README.md
 */
TEST_P(Medium_FusionDisabled, TrainsToTheSamePlaceAsFoldingIt)
{
    const std::vector<FusedForm> without{GetParam().form};
    FusionOptions narrowed;
    narrowed.disabled = SushiRuntime::span<const FusedForm>(without.data(), without.size());

    TrainedRun folded;
    ASSERT_NO_FATAL_FAILURE(train_with(GetParam().activation, FusionOptions{}, folded));
    TrainedRun separate;
    ASSERT_NO_FATAL_FAILURE(train_with(GetParam().activation, narrowed, separate));

    EXPECT_EQ(folded.fused_subtrees, separate.fused_subtrees)
        << "switching the three-node form off must fall back to gemm+bias, not to nothing";

    // Disabling the form costs exactly two kernels; see tests/layers/train/README.md.
    EXPECT_EQ(separate.operations_per_step, folded.operations_per_step + 2u);

    expect_runs_agree(folded, separate, "switching the fused epilogue off");
}

INSTANTIATE_TEST_SUITE_P(Forms, Medium_FusionDisabled,
                         ::testing::Values(DisabledRow{Activation::ReLU, FusedForm::GEMMBiasReLU,
                                                       "ReLU"},
                                           DisabledRow{Activation::GELU, FusedForm::GEMMBiasGELU,
                                                       "GELU"}),
                         disabled_name);

/**
 * @brief Tests that the fused ReLU MLP step issues ten fewer operations than the unfused one.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_Fusion, TheFusedStepIssuesTenFewerKernelsPerStep)
{
    TrainedRun fused;
    ASSERT_NO_FATAL_FAILURE(train_with(Activation::ReLU, FusionOptions{}, fused));
    TrainedRun plain;
    ASSERT_NO_FATAL_FAILURE(train_with(Activation::ReLU, off(), plain));

    EXPECT_EQ(plain.operations_per_step, fused.operations_per_step + 10u);
}

/**
 * @brief Tests that a fused GEMM depends on the task that writes the bias its epilogue reads.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_Fusion, AFusedGEMMIsOrderedAgainstTheTaskThatWritesItsBias)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_FEATURES}, "x");
    const Graph::ValueID w = builder.parameter(Shape{K_HIDDEN, K_FEATURES}, "w");
    const Graph::ValueID b = builder.parameter(Shape{K_HIDDEN}, "b");
    SA_EXPECT_OK(builder.relu(builder.bias_add(builder.matmul(x, w, false, true), b)));

    const Graph::FusionReport fusion = Graph::fuse(graph, FusionOptions{});
    ASSERT_EQ(1u, fusion.fused().size()) << fusion.to_string();

    TensorPool pool(engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

    SushiBLAS::LateCountHold late_counts(SaTest::TestEnvironment::runtime());
    SushiRuntime::Graph tasks = SaTest::TestEnvironment::runtime().graph();
    SushiBLAS::RuntimeScratchSource scratch_source(SaTest::TestEnvironment::runtime());
    SushiBLAS::ScratchPool scratch(scratch_source);
    SushiBLAS::TaskRecorder recorder(tasks.recorder(), scratch, late_counts, engine.mode());

    // Recorded first: the edge can only run to the GEMM; see tests/layers/train/README.md.
    SushiBLAS::Tensor& bias = pool.tensor(b);
    SushiBLAS::ElementwiseOps(recorder).mul(bias, bias, bias);

    const Graph::DeviceFusionSelector selector = Graph::DeviceFusionSelector::for_backend(backend);
    Graph::Lowering lowering(recorder, engine.rng(), pool, selector);
    ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
    ASSERT_EQ(1u, lowering.emitted_operations())
        << "this machine's device declined the fused form, so there is no epilogue to "
           "order and this test is not exercising what it claims to";

    SushiRuntime::Result<SushiRuntime::PlanView> plan = tasks.plan();
    ASSERT_TRUE(plan.has_value()) << plan.error().message;
    ASSERT_EQ(2u, plan->nodes.size());
    EXPECT_EQ(1u, plan->nodes[1].predecessor_count)
        << "The fused GEMM must depend on the task that wrote the bias its epilogue "
           "dereferences. A predecessor count of 0 means the read was never declared to "
           "SushiBLAS, and the optimizer's update is free to land before the forward "
           "pass has read the old value - which still trains, and still reports success.";
}

/**
 * @brief Tests that the reader of the spill depends on the fused GEMM that fills it.
 * @see tests/layers/train/README.md
 */
TEST_F(Medium_Fusion, WhateverReadsTheSpillIsOrderedBehindTheGEMMThatFillsIt)
{
    SushiBLAS::Engine engine(SaTest::TestEnvironment::runtime());
    SushiAI::RuntimeExecutionBackend backend(SaTest::TestEnvironment::runtime(), engine);

    Graph::Graph graph;
    const Graph::GraphBuilder builder(graph);
    const Graph::ValueID x = builder.input(Shape{K_BATCH, K_FEATURES}, "x");
    const Graph::ValueID w = builder.parameter(Shape{K_HIDDEN, K_FEATURES}, "w");
    const Graph::ValueID b = builder.parameter(Shape{K_HIDDEN}, "b");
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID pre,
                            builder.bias_add(builder.matmul(x, w, false, true), b));
    SA_ASSERT_OK_AND_ASSIGN(const Graph::ValueID out, builder.gelu(pre));

    // Emitted after the GELU, as gelu_backward is; see tests/layers/train/README.md.
    const Graph::ValueID dy = builder.input(Shape{K_BATCH, K_HIDDEN}, "dy");
    SA_EXPECT_OK(builder.activation_backward(Ops::K_GELU_BACKWARD.id, dy, pre));

    const std::vector<Graph::ValueID> observed{out};
    FusionOptions options;
    options.observed = SushiRuntime::span<const Graph::ValueID>(observed.data(), observed.size());

    const Graph::FusionReport fusion = Graph::fuse(graph, options);
    ASSERT_EQ(1u, fusion.count(FusedForm::GEMMBiasGELU)) << fusion.to_string();

    TensorPool pool(engine);
    ASSERT_TRUE(SaTest::status_is_ok(pool.materialise_leaves(graph)));

    SushiBLAS::LateCountHold late_counts(SaTest::TestEnvironment::runtime());
    SushiRuntime::Graph tasks = SaTest::TestEnvironment::runtime().graph();
    SushiBLAS::RuntimeScratchSource scratch_source(SaTest::TestEnvironment::runtime());
    SushiBLAS::ScratchPool scratch(scratch_source);
    SushiBLAS::TaskRecorder recorder(tasks.recorder(), scratch, late_counts, engine.mode());

    const Graph::DeviceFusionSelector selector = Graph::DeviceFusionSelector::for_backend(backend);
    Graph::Lowering lowering(recorder, engine.rng(), pool, selector);
    ASSERT_TRUE(SaTest::status_is_ok(lowering.record(graph)));
    ASSERT_EQ(2u, lowering.emitted_operations())
        << "expected the fused GEMM and the activation gradient; this machine's device may "
           "have declined the fused form, in which case this test is not exercising what "
           "it claims to";

    SushiRuntime::Result<SushiRuntime::PlanView> plan = tasks.plan();
    ASSERT_TRUE(plan.has_value()) << plan.error().message;
    ASSERT_EQ(2u, plan->nodes.size());
    EXPECT_EQ(1u, plan->nodes[1].predecessor_count)
        << "The task that reads the spilled pre-activation must depend on the fused GEMM "
           "whose epilogue wrote it. A predecessor count of 0 means `.writes(aux)` was "
           "never declared, the two may run concurrently, and the spill buys nothing at "
           "all - while the run still trains and still reports success.";
}

/** @brief Tests that a described device without fp64 is refused a fused double kernel. */
TEST_F(Small_FusionSelector, ADeviceWithoutDoublePrecisionIsNotHandedAFusedDoubleKernel)
{
    const Graph::DeviceFusionSelector without(pretend_device(false),
                                              SushiBLAS::ExecutionMode::Auto);
    const Graph::DeviceFusionSelector with(pretend_device(true),
                                           SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBiasReLU;
    query.dtype = DType::Float64;
    query.rows = 32;
    query.columns = 128;

    EXPECT_EQ(without.choose(query).form, FusedForm::None);
    EXPECT_EQ(with.choose(query).form, FusedForm::GEMMBiasReLU);

    // A float graph is fine on either.
    query.dtype = DType::Float32;
    EXPECT_EQ(without.choose(query).form, FusedForm::GEMMBiasReLU);
}

/**
 * @brief Tests that the host-executing mode is refused a GEMM epilogue and keeps the zip form.
 * @see tests/layers/train/README.md
 */
TEST_F(Small_FusionSelector, AHostExecutingModeDoesNotGetAnEpilogueThatDereferencesDeviceMemory)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::ForceReference);

    Graph::FusionQuery gemm;
    gemm.matched = FusedForm::GEMMBias;
    gemm.rows = 32;
    gemm.columns = 128;
    EXPECT_EQ(selector.choose(gemm).form, FusedForm::None);

    // The zip form runs no host arm; see tests/layers/train/README.md.
    Graph::FusionQuery zip;
    zip.matched = FusedForm::ExpSubScale;
    zip.rows = 32;
    zip.columns = 128;
    EXPECT_EQ(selector.choose(zip).form, FusedForm::ExpSubScale);
}

/** @brief Tests that a bias that is not contiguous gets no epilogue that indexes it directly. */
TEST_F(Small_FusionSelector, AStridedBiasIsNotHandedToAnEpilogueThatIndexesItDirectly)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBias;
    query.rows = 32;
    query.columns = 128;
    query.operand_is_contiguous = false;

    EXPECT_EQ(selector.choose(query).form, FusedForm::None);
}

/**
 * @brief Tests that every selector choice carries a non-empty reason.
 * @see tests/layers/train/README.md
 */
TEST_F(Small_FusionSelector, EveryChoiceCarriesAReasonSomebodyCanRead)
{
    const Graph::DeviceFusionSelector selector(pretend_device(true),
                                               SushiBLAS::ExecutionMode::Auto);

    Graph::FusionQuery query;
    query.matched = FusedForm::GEMMBiasReLU;
    query.rows = 32;
    query.columns = 128;

    EXPECT_FALSE(selector.choose(query).reason.empty());

    query.operand_is_contiguous = false;
    EXPECT_FALSE(selector.choose(query).reason.empty());
}
