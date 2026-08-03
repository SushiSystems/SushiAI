/**************************************************************************/
/* test_resilience.cpp                                                    */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/**************************************************************************/

// What the library does when it is handed something it cannot serve, and what
// it does at the edges of what it can.
//
// Three separate claims live here, and none of them is "the happy path
// works" - that is the job of test_forward.cpp and test_training.cpp:
//
//  1. Every device-lane precondition fails loudly AND names the actual
//     problem. See tests/common/test_throws.hpp for why the message is
//     asserted rather than just the throw.
//  2. Degenerate and boundary extents - one sample, one feature, one class,
//     exactly the GEMM register-block edge and exactly one past it - produce
//     right answers rather than off-by-one ones. Index arithmetic breaks here
//     or nowhere.
//  3. Numbers that would make a naive implementation produce NaN do not. A
//     NaN is the worst failure this library has: it propagates into every
//     weight on the next update, the loss stops moving, and nothing reports
//     an error. Finiteness is therefore asserted explicitly rather than
//     inferred from a loss that looks plausible.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include "../../common/test_throws.hpp"
#include "../../common/test_utils.hpp"

using namespace SushiAI;

// `Test` alone resolves to ::testing::Test inside a TEST_F body, where the
// fixture's base injects the name; the alias keeps the helpers reachable.
namespace SaTest = SushiAI::Test;
using Graph::CompiledStep;
using Graph::GraphBuilder;
using Graph::TensorPool;
using Graph::ValueId;

namespace
{
    class Integration_Resilience : public SaTest::EngineTest {};

    using Mlp = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    /**
     * @class Runner
     * @brief Materialises a graph's leaves, compiles once, and replays.
     *
     * The same shape as test_forward.cpp's OneShot, kept local rather than
     * shared because the two files probe different things and a shared
     * fixture would couple their evolution.
     */
    class Runner
    {
        public:
            Runner(SushiBLAS::Engine& engine, const Graph::Graph& graph)
                : pool(engine), graph_(&graph), engine_(&engine)
            {
                pool.materialise_leaves(graph);
            }

            /** @brief Compiles the graph; call once every leaf has been written. */
            void compile() { step_ = std::make_unique<CompiledStep>(*engine_, *graph_, pool); }

            /** @brief Replays the compiled step. */
            void run() { step_->run(); }

            TensorPool pool;

        private:
            const Graph::Graph* graph_;
            SushiBLAS::Engine* engine_;
            std::unique_ptr<CompiledStep> step_;
    };

    /**
     * @class NoBatches
     * @brief A dataset with the right shapes and nothing in it.
     *
     * SyntheticClassification refuses to generate zero batches, so an empty
     * epoch cannot be reached through it - which is exactly why the trainer's
     * own guard needs a dataset that can.
     */
    class NoBatches final : public Data::Dataset
    {
        public:
            NoBatches(int64_t batch, int64_t features, int64_t classes) noexcept
                : batch_(batch), features_(features), classes_(classes) {}

            int64_t batch_size() const noexcept override { return batch_; }
            int64_t input_dim() const noexcept override { return features_; }
            int64_t num_classes() const noexcept override { return classes_; }
            std::size_t batch_count() const noexcept override { return 0; }
            std::string_view name() const noexcept override { return "no-batches"; }

            const Data::Batch& batch(std::size_t index) const override
            {
                SA_THROW_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return never_;
            }

        private:
            int64_t batch_ = 0;
            int64_t features_ = 0;
            int64_t classes_ = 0;
            Data::Batch never_{};
    };

    /** @brief Every element of @p values is finite. */
    ::testing::AssertionResult all_finite(const std::vector<double>& values,
                                          std::string_view what)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
            if (!std::isfinite(values[i]))
                return ::testing::AssertionFailure()
                    << what << "[" << i << "] is " << values[i]
                    << " - a non-finite value propagates into every weight on the next "
                       "update and the loss simply stops moving";

        return ::testing::AssertionSuccess();
    }

    /** @brief A ready-to-train MLP with its pool, optimizer and trainer. */
    struct Fixture
    {
        Mlp model;
        Train::TrainingGraph built;
        std::unique_ptr<TensorPool> pool;
        std::vector<Optim::ParameterSlot> slots;
    };

    /** @brief Traces and binds an MLP of the given extents. */
    Fixture make_fixture(SushiBLAS::Engine& engine, int64_t batch, int64_t features,
                         int64_t hidden, int64_t classes)
    {
        Fixture fixture{Mlp(NN::Linear{features, hidden}, NN::ReLU{},
                            NN::Linear{hidden, classes}), {}, nullptr, {}};

        fixture.built = Train::build_classifier(fixture.model, batch, features, classes);
        fixture.pool = std::make_unique<TensorPool>(engine);
        fixture.slots = Train::bind_parameters(fixture.built, *fixture.pool);
        NN::initialize(engine, fixture.model, *fixture.pool);
        return fixture;
    }

    /** @brief A synthetic problem of the given extents. */
    Data::SyntheticOptions data_for(int64_t batch, int64_t batches, int64_t features,
                                    int64_t classes)
    {
        Data::SyntheticOptions options;
        options.batch_size = batch;
        options.batches = batches;
        options.features = features;
        options.classes = classes;
        return options;
    }
}

// =========================================================================
// 1. Loud failures that name the problem
// =========================================================================

TEST_F(Integration_Resilience, AnOptimizerNamesTheParameterWhoseGradientDoesNotMatch)
{
    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight",
                                         allocate_tensor(*engine, Shape{4, 3}),
                                         allocate_tensor(*engine, Shape{4, 2})});

    SA_EXPECT_ERROR((void)Optim::SGD(*engine, slots, Optim::SGDOptions{}),
                    "0.weight", "12 elements", "gradient has 8");
}

TEST_F(Integration_Resilience, AnOptimizerNamesBothDtypesWhenAGradientIsTheWrongType)
{
    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.bias",
                                         allocate_tensor(*engine, Shape{4}, DType::FLOAT32),
                                         allocate_tensor(*engine, Shape{4}, DType::FLOAT64)});

    SA_EXPECT_ERROR((void)Optim::SGD(*engine, slots, Optim::SGDOptions{}),
                    "0.bias", "float32", "float64");
}

TEST_F(Integration_Resilience, AnOptimizerRefusesADtypeItCannotServeAndExplainsWhy)
{
    // Complex is not a training dtype at all; half is a real float but is
    // refused for a reason the message has to carry, because "it compiles and
    // trains worse" is precisely the outcome the refusal exists to prevent.
    std::vector<Optim::ParameterSlot> complex_slot;
    complex_slot.push_back(Optim::ParameterSlot{"0.weight",
                                                allocate_tensor(*engine, Shape{4}, DType::COMPLEX32),
                                                allocate_tensor(*engine, Shape{4}, DType::COMPLEX32)});

    SA_EXPECT_ERROR((void)Optim::SGD(*engine, complex_slot, Optim::SGDOptions{}),
                    "0.weight", "complex32", "real floating tensors only");

    std::vector<Optim::ParameterSlot> half_slot;
    half_slot.push_back(Optim::ParameterSlot{"0.weight",
                                             allocate_tensor(*engine, Shape{4}, DType::HALF),
                                             allocate_tensor(*engine, Shape{4}, DType::HALF)});

    SA_EXPECT_ERROR((void)Optim::AdamW(*engine, half_slot, Optim::AdamWOptions{}),
                    "0.weight", "half precision", "float32");
}

TEST_F(Integration_Resilience, AdamWNamesTheHyperparameterThatIsOutOfRange)
{
    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"0.weight",
                                         allocate_tensor(*engine, Shape{4}),
                                         allocate_tensor(*engine, Shape{4})});

    Optim::AdamWOptions bad_beta1;
    bad_beta1.beta1 = 1.0;
    SA_EXPECT_ERROR((void)Optim::AdamW(*engine, slots, bad_beta1), "beta1", "[0, 1)");

    Optim::AdamWOptions bad_beta2;
    bad_beta2.beta2 = -0.5;
    SA_EXPECT_ERROR((void)Optim::AdamW(*engine, slots, bad_beta2), "beta2", "[0, 1)");

    Optim::AdamWOptions bad_eps;
    bad_eps.eps = 0.0;
    SA_EXPECT_ERROR((void)Optim::AdamW(*engine, slots, bad_eps), "eps must be positive");
}

TEST_F(Integration_Resilience, ADisconnectedParameterNamesItselfAndSaysWhatToDo)
{
    NN::Linear layer{4, 3};
    Train::TrainingGraph built = Train::build_classifier(layer, 4, 4, 3);
    built.parameters.push_back(Train::ParameterBinding{"ghost", layer.weight.id(),
                                                       Graph::NO_VALUE});

    TensorPool pool(*engine);
    SA_EXPECT_ERROR((void)Train::bind_parameters(built, pool),
                    "'ghost'", "receives no gradient", "no optimizer could ever move it");
}

TEST_F(Integration_Resilience, AnOptimizerStepBeforeAnyBackwardIsANoOpRatherThanGarbage)
{
    // There is no zero_grad() in this library, by design: every gradient
    // buffer in a step is written rather than accumulated into (see
    // optimizer.hpp). That leaves exactly one window in which a gradient has
    // never been written at all - between bind_parameters() and the first
    // replay - and stepping in that window would otherwise feed whatever the
    // allocator last had in that memory straight into the weights. There is
    // nothing to catch it downstream: the parameters simply become garbage or
    // NaN, and the loss never recovers.
    Fixture fixture = make_fixture(*engine, 8, 6, 12, 4);

    for (const Optim::ParameterSlot& slot : fixture.slots)
    {
        const std::vector<double> gradient = SaTest::read_tensor(slot.gradient);
        ASSERT_TRUE(all_finite(gradient, slot.name + ".grad"));

        for (std::size_t i = 0; i < gradient.size(); ++i)
            ASSERT_EQ(gradient[i], 0.0)
                << "'" << slot.name << "' gradient element " << i
                << " holds " << gradient[i] << " before any backward pass has run";
    }

    const std::vector<std::vector<double>> before = [&fixture]()
    {
        std::vector<std::vector<double>> snapshot;
        for (const Optim::ParameterSlot& slot : fixture.slots)
            snapshot.push_back(SaTest::read_tensor(slot.parameter));
        return snapshot;
    }();

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.1;
    Optim::SGD optimizer(*engine, fixture.slots, sgd);
    optimizer.step();

    for (std::size_t p = 0; p < fixture.slots.size(); ++p)
    {
        const std::vector<double> after = SaTest::read_tensor(fixture.slots[p].parameter);
        ASSERT_EQ(after.size(), before[p].size());

        for (std::size_t i = 0; i < after.size(); ++i)
            EXPECT_EQ(after[i], before[p][i])
                << "'" << fixture.slots[p].name << "' moved on a step with no gradient, at " << i;
    }
}

TEST_F(Integration_Resilience, TheTrainerNamesTheBatchThatDoesNotFitTheCompiledStep)
{
    Fixture fixture = make_fixture(*engine, 8, 6, 10, 4);
    Optim::SGD optimizer(*engine, fixture.slots, Optim::SGDOptions{});
    Train::Trainer trainer(*engine, fixture.built.graph, *fixture.pool,
                           fixture.built.spec, optimizer);

    // A step's buffers are addresses baked into a compiled plan, so a batch of
    // the wrong size cannot be accommodated - only reported.
    const Data::SyntheticClassification wrong(*engine, data_for(4, 1, 6, 4));

    SA_EXPECT_ERROR((void)trainer.run_step(wrong.batch(0)),
                    "24 input elements", "expects 48");
}

TEST_F(Integration_Resilience, FitNamesTheDatasetDimensionThatDisagrees)
{
    Fixture fixture = make_fixture(*engine, 8, 6, 10, 4);
    Optim::SGD optimizer(*engine, fixture.slots, Optim::SGDOptions{});
    Train::Trainer trainer(*engine, fixture.built.graph, *fixture.pool,
                           fixture.built.spec, optimizer);

    const NoBatches empty(8, 6, 4);
    SA_EXPECT_ERROR((void)trainer.fit(empty), "The dataset holds no batches");

    const Data::SyntheticClassification narrow(*engine, data_for(4, 1, 6, 4));
    SA_EXPECT_ERROR((void)trainer.fit(narrow), "batch size is 4", "compiled for 8");

    const Data::SyntheticClassification wrong_classes(*engine, data_for(8, 1, 6, 5));
    SA_EXPECT_ERROR((void)trainer.fit(wrong_classes), "5 classes", "compiled for 4");
}

TEST_F(Integration_Resilience, TheSyntheticDatasetNamesEveryExtentWhenOneIsNotPositive)
{
    SA_EXPECT_ERROR((void)Data::SyntheticClassification(*engine, data_for(8, 0, 6, 4)),
                    "positive extents", "batches=0");
    SA_EXPECT_ERROR((void)Data::SyntheticClassification(*engine, data_for(0, 2, 6, 4)),
                    "positive extents", "batch=0");
    SA_EXPECT_ERROR((void)Data::SyntheticClassification(*engine, data_for(8, 2, 6, 0)),
                    "positive extents", "classes=0");
}

TEST_F(Integration_Resilience, AllocationRefusesAnEmptyOrUnrepresentableTensor)
{
    const int64_t empty_dims[2] = {4, 0};
    SA_EXPECT_ERROR((void)allocate_tensor(*engine, SushiRuntime::span<const int64_t>(empty_dims, 2)),
                    "Tensor extents must be positive", "got 0");

    // Signed overflow in the extent product is undefined behaviour, which a
    // compiler may assume away along with any check written after it. Detected
    // before the multiply, so the diagnostic is about the shape rather than
    // about the wreckage.
    const int64_t overflowing[3] = {3'000'000'000LL, 3'000'000'000LL, 3'000'000'000LL};
    SA_EXPECT_ERROR((void)allocate_tensor(*engine, SushiRuntime::span<const int64_t>(overflowing, 3)),
                    "more than", "elements", "3000000000");
}

TEST_F(Integration_Resilience, AModelTooLargeToAllocateNamesTheTensorRatherThanTheKernel)
{
    // A petabyte. No allocator serves it, and the point is what the caller is
    // told: the shape and element type it asked for, not a bare byte count
    // from four frames down, and not a null pointer for a kernel to chase.
    const int64_t enormous[2] = {1LL << 25, 1LL << 25};

    SA_EXPECT_ERROR((void)allocate_tensor(*engine, SushiRuntime::span<const int64_t>(enormous, 2)),
                    "Could not allocate the float32 tensor", "[33554432, 33554432]");
}

// =========================================================================
// 2. Degenerate and boundary extents
// =========================================================================

TEST_F(Integration_Resilience, ASingleSampleWithASingleFeatureTrains)
{
    // N = 1, D = 1, and one Linear with no hidden layer. Every loop bound in
    // the lowering is at its minimum simultaneously.
    const int64_t batch = 1;
    const int64_t features = 1;
    const int64_t classes = 3;

    NN::Linear layer{features, classes};
    Train::TrainingGraph built = Train::build_classifier(layer, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);
    NN::initialize(*engine, layer, pool);

    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.5;
    Optim::SGD optimizer(*engine, slots, sgd);

    Train::TrainerOptions options;
    options.track_accuracy = false;
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer, options);

    const Data::SyntheticClassification dataset(*engine, data_for(batch, 2, features, classes));

    const Train::StepMetrics first = trainer.run_step(dataset.batch(0));
    EXPECT_NEAR(first.loss, std::log(static_cast<double>(classes)), 1e-5);

    for (int step = 0; step < 8; ++step)
    {
        const Train::StepMetrics metrics = trainer.run_step(dataset.batch(static_cast<std::size_t>(step % 2)));
        ASSERT_TRUE(std::isfinite(metrics.loss)) << "at step " << step;
    }

    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].gradient), "gradient"));
}

TEST_F(Integration_Resilience, ASingleClassIsDegenerateButNotUndefined)
{
    // With C = 1 the softmax is identically one, so the loss is exactly zero
    // and the gradient (p - y) / N is exactly zero. Nothing learns, which is
    // correct - but 0/0 anywhere in the recipe would show up as NaN instead.
    const int64_t batch = 4;
    const int64_t features = 3;
    const int64_t classes = 1;

    NN::Linear layer{features, classes};
    Train::TrainingGraph built = Train::build_classifier(layer, batch, features, classes);

    TensorPool pool(*engine);
    std::vector<Optim::ParameterSlot> slots = Train::bind_parameters(built, pool);

    NN::InitPolicy policy;
    policy.zero_final_layer = false;
    NN::initialize(*engine, layer, pool, policy);

    Optim::AdamW optimizer(*engine, slots, Optim::AdamWOptions{});
    Train::Trainer trainer(*engine, built.graph, pool, built.spec, optimizer);

    const Data::SyntheticClassification dataset(*engine, data_for(batch, 1, features, classes));

    const Train::StepMetrics metrics = trainer.run_step(dataset.batch(0));

    EXPECT_NEAR(metrics.loss, 0.0, 1e-4);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 1.0);
    EXPECT_TRUE(all_finite(SaTest::read_tensor(slots[0].parameter), "weight"));

    const std::vector<double> gradient = SaTest::read_tensor(slots[0].gradient);
    EXPECT_TRUE(all_finite(gradient, "gradient"));
    for (std::size_t i = 0; i < gradient.size(); ++i)
        EXPECT_NEAR(gradient[i], 0.0, 1e-4) << "at " << i;
}

TEST_F(Integration_Resilience, MatmulIsExactAtTheRegisterBlockEdgeAndOnePastIt)
{
    // The local-tiled GEMM stages a 64x64 block of float per work-group and
    // pads the ragged remainder with zeros. An extent of exactly 64 exercises
    // the padding-free path; 65 leaves a one-row, one-column remainder, which
    // is where an off-by-one in the boundary test shows up as a wrong answer
    // rather than as a fault. Both are checked against a host reference, and
    // with the transpose FLAG the Linear layer actually emits.
    for (int64_t n : {int64_t{64}, int64_t{65}})
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{n, n}, "x");
        const ValueId w = builder.input(Shape{n, n}, "w");
        const ValueId y = builder.matmul(x, w, false, true);

        Runner run(*engine, graph);

        std::vector<double> x_values(static_cast<std::size_t>(n * n));
        std::vector<double> w_values(static_cast<std::size_t>(n * n));
        SaTest::HostNoise(0xB10Cu + static_cast<std::uint64_t>(n)).fill(x_values, 1.0);
        SaTest::HostNoise(0xB10Du + static_cast<std::uint64_t>(n)).fill(w_values, 1.0);

        SaTest::write_tensor(run.pool.tensor(x), x_values);
        SaTest::write_tensor(run.pool.tensor(w), w_values);

        run.compile();
        run.run();

        const std::vector<double> got = SaTest::read_tensor(run.pool.tensor(y));
        ASSERT_EQ(got.size(), static_cast<std::size_t>(n * n));

        // The values that reached the device are float, so the reference has
        // to be computed from those same floats or the comparison measures
        // rounding rather than indexing.
        const std::vector<double> x_seen = SaTest::read_tensor(run.pool.tensor(x));
        const std::vector<double> w_seen = SaTest::read_tensor(run.pool.tensor(w));

        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                double expected = 0.0;
                for (int64_t k = 0; k < n; ++k)
                    expected += x_seen[static_cast<std::size_t>(i * n + k)]
                              * w_seen[static_cast<std::size_t>(j * n + k)];

                const double actual = got[static_cast<std::size_t>(i * n + j)];
                ASSERT_NEAR(actual, expected, 2e-3)
                    << "n=" << n << " at (" << i << ", " << j << ")";
            }
        }
    }
}

TEST_F(Integration_Resilience, TrainingIsFiniteAtTheRegisterBlockEdgeAndOnePastIt)
{
    for (int64_t n : {int64_t{64}, int64_t{65}})
    {
        Fixture fixture = make_fixture(*engine, n, n, n, 4);
        Optim::SGDOptions sgd;
        sgd.learning_rate = 0.1;
        Optim::SGD optimizer(*engine, fixture.slots, sgd);

        Train::TrainerOptions options;
        options.track_accuracy = false;
        Train::Trainer trainer(*engine, fixture.built.graph, *fixture.pool,
                               fixture.built.spec, optimizer, options);

        const Data::SyntheticClassification dataset(*engine, data_for(n, 2, n, 4));

        for (std::size_t step = 0; step < 4; ++step)
        {
            const Train::StepMetrics metrics = trainer.run_step(dataset.batch(step % 2));
            ASSERT_TRUE(std::isfinite(metrics.loss)) << "n=" << n << " step " << step;
        }

        for (const Optim::ParameterSlot& slot : fixture.slots)
        {
            EXPECT_TRUE(all_finite(SaTest::read_tensor(slot.parameter), slot.name))
                << "n=" << n;
            EXPECT_TRUE(all_finite(SaTest::read_tensor(slot.gradient), slot.name + ".grad"))
                << "n=" << n;
        }
    }
}

// =========================================================================
// 3. Numerical extremes
// =========================================================================

namespace
{
    /**
     * @brief Log-softmax of @p logits, read back, for the stability checks.
     *
     * @param engine The engine to run on.
     * @param rows Rows of the logit matrix.
     * @param columns Columns of the logit matrix.
     * @param logits The values to feed, row-major.
     * @return The [rows, columns] log-probabilities.
     */
    std::vector<double> log_softmax_of(SushiBLAS::Engine& engine, int64_t rows, int64_t columns,
                                       const std::vector<double>& logits)
    {
        Graph::Graph graph;
        const GraphBuilder builder(graph);

        const ValueId x = builder.input(Shape{rows, columns}, "logits");
        const ValueId y = builder.log_softmax(x);

        Runner run(engine, graph);
        SaTest::write_tensor(run.pool.tensor(x), logits);
        run.compile();
        run.run();

        return SaTest::read_tensor(run.pool.tensor(y));
    }
}

TEST_F(Integration_Resilience, LogSoftmaxOfUniformLogitsIsMinusLogCAtAnyMagnitude)
{
    // The max shift is the only reason any of these is representable: without
    // it exp(1e30) is inf and exp(-1e30) is zero, and both give NaN. With it,
    // every one of them must land on the same exact answer, because a uniform
    // row is uniform whatever its offset.
    const int64_t rows = 3;
    const int64_t columns = 4;
    const double expected = -std::log(static_cast<double>(columns));

    for (double magnitude : {0.0, 1.0, -1.0, 1e30, -1e30, 3.0e38})
    {
        const std::vector<double> logits(static_cast<std::size_t>(rows * columns), magnitude);
        const std::vector<double> got = log_softmax_of(*engine, rows, columns, logits);

        ASSERT_TRUE(all_finite(got, "log_softmax")) << "at magnitude " << magnitude;
        for (std::size_t i = 0; i < got.size(); ++i)
            EXPECT_NEAR(got[i], expected, 1e-5) << "magnitude " << magnitude << " at " << i;
    }
}

TEST_F(Integration_Resilience, LogSoftmaxSurvivesBothExtremesInsideOneRow)
{
    // +1e30 against -1e30 in the same row: the shift drives the winner to
    // exactly zero and every loser to -2e30, whose exponential underflows to
    // zero without ever becoming a NaN. The row sum is therefore exactly one,
    // and log(1) is exactly zero, so the winner's log-probability is exactly
    // zero - a probability of one, which is the right answer for a logit gap
    // of 2e30.
    const int64_t rows = 2;
    const int64_t columns = 3;

    std::vector<double> logits = {
        1e30, -1e30, -1e30,
        -1e30, 1e30, -1e30,
    };

    const std::vector<double> got = log_softmax_of(*engine, rows, columns, logits);

    ASSERT_TRUE(all_finite(got, "log_softmax"));
    EXPECT_NEAR(got[0], 0.0, 1e-6);
    EXPECT_NEAR(got[4], 0.0, 1e-6);
    EXPECT_LT(got[1], -1e29);
    EXPECT_LT(got[3], -1e29);
}

TEST_F(Integration_Resilience, LogSoftmaxIsExactAcrossRowsThatSpanTheExponentRange)
{
    // A gap of 60 between the two rows' own maxima. This used to pin the
    // *supported side* of a boundary: the shift was by the batch's global
    // maximum, so a row far enough below it came back +inf, and 60 was
    // comfortably inside the range that still worked.
    //
    // There is no boundary now - the shift is each row's own maximum, so the
    // gap does not reach the arithmetic - and the case is kept unchanged
    // anyway. It is the shape a training run actually produces, and a
    // regression that reintroduced a global shift would still pass every test
    // whose rows sit on top of each other. The gaps that break a global shift
    // are pinned separately, with their measured errors, in
    // tests/functional/regression/test_log_softmax_stability.cpp.
    const int64_t columns = 3;

    std::vector<double> logits = {
        60.0, 59.0, 58.0,   // the batch maximum
        0.0, -1.0, -2.0,    // 60 below it, still exact
    };

    const std::vector<double> got = log_softmax_of(*engine, 2, columns, logits);
    ASSERT_TRUE(all_finite(got, "log_softmax"));

    // Both rows have the same internal spread, so both must produce the same
    // log-probabilities - which is the whole claim of shift invariance.
    for (std::size_t column = 0; column < 3; ++column)
        EXPECT_NEAR(got[column], got[column + 3], 1e-4) << "column " << column;

    const double denominator = 1.0 + std::exp(-1.0) + std::exp(-2.0);
    EXPECT_NEAR(got[0], -std::log(denominator), 1e-4);
}

TEST_F(Integration_Resilience, CrossEntropyIsFiniteOnPerfectlyConfidentPredictions)
{
    const int64_t rows = 2;
    const int64_t columns = 3;

    Graph::Graph graph;
    const GraphBuilder builder(graph);

    const ValueId logits = builder.input(Shape{rows, columns}, "logits");
    const ValueId targets = builder.input(Shape{rows, columns}, "targets");
    const ValueId loss = NN::CrossEntropyLoss{}.forward(builder, logits, targets);

    Runner run(*engine, graph);

    // Confidence of exp(80) either way. Anything that exponentiates without
    // shifting first overflows here; anything that takes a log of a rounded
    // zero produces -inf.
    const std::vector<double> confident = {
        80.0, -80.0, -80.0,
        -80.0, 80.0, -80.0,
    };
    SaTest::write_tensor(run.pool.tensor(logits), confident);
    SaTest::write_tensor(run.pool.tensor(targets),
                         std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    run.compile();
    run.run();

    const double correct = SaTest::read_tensor(run.pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(correct));
    EXPECT_NEAR(correct, 0.0, 1e-5) << "a perfectly confident correct prediction costs nothing";

    // The same logits against the opposite targets: the loss is large, and
    // large is the correct answer. It must not be inf, because an infinite
    // loss makes every gradient below it a NaN.
    SaTest::write_tensor(run.pool.tensor(targets),
                         std::vector<double>{0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    run.run();

    const double wrong = SaTest::read_tensor(run.pool.tensor(loss))[0];
    EXPECT_TRUE(std::isfinite(wrong)) << "loss on a confidently wrong prediction";
    EXPECT_NEAR(wrong, 160.0, 1e-2);
}

TEST_F(Integration_Resilience, AdamWsFirstStepWithAZeroGradientStaysFinite)
{
    // m and v are both exactly zero on step one, so the update is
    // 0 / (sqrt(0 / bc2) + eps). It is eps in the denominator, and only eps,
    // that keeps that a zero rather than a NaN - and eps is added AFTER the
    // bias correction, so it is not scaled away. Fold it inside and the
    // denominator becomes sqrt(0 + eps) / sqrt(bc2), which is still non-zero
    // here but wrong everywhere else; drop it and this is 0/0.
    SushiBLAS::Tensor parameter = allocate_tensor(*engine, Shape{6});
    SushiBLAS::Tensor gradient = allocate_tensor(*engine, Shape{6});

    const std::vector<double> initial = {0.5, -1.5, 0.0, 2.0, -0.25, 100.0};
    SaTest::write_tensor(parameter, initial);
    SaTest::write_tensor(gradient, std::vector<double>(6, 0.0));

    std::vector<Optim::ParameterSlot> slots;
    slots.push_back(Optim::ParameterSlot{"p", parameter, gradient});

    Optim::AdamWOptions options;
    options.learning_rate = 0.1;
    options.weight_decay = 0.0;
    Optim::AdamW optimizer(*engine, slots, options);

    optimizer.step();

    const std::vector<double> after = SaTest::read_tensor(parameter);
    ASSERT_TRUE(all_finite(after, "parameter"));

    // With no decay and no gradient there is nothing to move by, so the
    // parameter must be untouched rather than merely finite.
    for (std::size_t i = 0; i < initial.size(); ++i)
        EXPECT_NEAR(after[i], initial[i], 1e-6) << "at " << i;

    // A second step still has nothing to work with, and the bias corrections
    // have changed - which is where a divide that only happened to be safe on
    // step one would come apart.
    optimizer.step();
    EXPECT_TRUE(all_finite(SaTest::read_tensor(parameter), "parameter after two steps"));
}

// =========================================================================
// 4. Resource discipline
// =========================================================================

TEST_F(Integration_Resilience, BuildingAndDestroyingTrainersReachesASteadyState)
{
    // Every SushiBLAS::Storage holds a counted reference to the context's
    // allocator, so the allocator's reference count IS the number of live
    // allocations plus a constant. That makes "did the last iteration leak a
    // buffer" a number this test can read rather than something to eyeball in
    // Task Manager - one leaked Storage moves it by exactly one.
    //
    // The baseline is taken after the FIRST iteration, not before it, so a
    // one-time lazy cache inside the runtime is tolerated while a per-
    // iteration leak is not.
    const auto live_allocations = [this]()
    {
        return engine->get_context().get_allocator()->get_ref_count();
    };

    std::int32_t baseline = 0;
    std::size_t baseline_bytes = 0;

    for (int iteration = 0; iteration < 5; ++iteration)
    {
        {
            Fixture fixture = make_fixture(*engine, 8, 6, 12, 4);
            Optim::AdamW optimizer(*engine, fixture.slots, Optim::AdamWOptions{});

            Train::TrainerOptions options;
            options.track_accuracy = false;
            Train::Trainer trainer(*engine, fixture.built.graph, *fixture.pool,
                                   fixture.built.spec, optimizer, options);

            const Data::SyntheticClassification dataset(*engine, data_for(8, 2, 6, 4));
            (void)trainer.fit(dataset);

            if (iteration == 0) baseline_bytes = fixture.pool->allocated_bytes();
            EXPECT_EQ(fixture.pool->allocated_bytes(), baseline_bytes)
                << "the same model allocated a different amount on iteration " << iteration;
        }

        engine->get_context().wait_all();

        if (iteration == 0) baseline = live_allocations();
        else
            EXPECT_EQ(live_allocations(), baseline)
                << "iteration " << iteration << " left "
                << (live_allocations() - baseline)
                << " allocation(s) behind; a trainer's buffers must all die with it";
    }
}
