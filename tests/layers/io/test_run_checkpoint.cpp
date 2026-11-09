/**************************************************************************/
/* test_run_checkpoint.cpp                                                */
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
 * @file test_run_checkpoint.cpp
 * @brief Tests that a run resumed from a checkpoint with run state continues bit for bit.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;
// SaTest stays reachable inside a TEST_F body; see tests/layers/io/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    class Medium_RunCheckpoint : public SaTest::EngineTest {};

    using MLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Linear>;

    constexpr int64_t K_BATCH = 8;
    constexpr int64_t K_FEATURES = 6;
    constexpr int64_t K_CLASSES = 4;
    constexpr int64_t K_BATCHES = 4;

    /** @brief Holds the updates before the save, and as many again after the restore. */
    constexpr std::uint64_t K_HALF_UPDATES = 6;

    /** @brief Holds the momentum of the SGD runs that keep a buffer. */
    constexpr double K_MOMENTUM = 0.9;

    /** @brief Holds the offset of the run-state count in a version 2 header. */
    constexpr std::size_t K_OFFSET_RUN_STATE_COUNT = 12;

    /** @brief Names the optimizer a run trains with. */
    enum class Kind : std::uint8_t
    {
        SGD,   /**< SGD, with the momentum the run states. */
        AdamW, /**< AdamW with weight decay. */
    };

    /**
     * @struct TrainedRun
     * @brief Holds a model, its state, an optimizer and the compiled step over them.
     */
    struct TrainedRun
    {
        /** @brief Builds the pool on @p engine; make_run() fills the rest. */
        explicit TrainedRun(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief The model the graph is traced from. */
        MLP model = MLP(NN::Linear{K_FEATURES, 12}, NN::ReLU{}, NN::Linear{12, K_CLASSES});

        /** @brief The training graph of the model. */
        std::optional<Train::TrainingGraph> built;

        /** @brief The parameters of the model. */
        std::optional<NN::ModelState> state;

        /** @brief The pool the graph's tensors live in. */
        Graph::TensorPool pool;

        /** @brief The objective the graph was traced with. */
        std::optional<Train::CrossEntropyObjective> objective;

        /** @brief The optimizer over the parameters of the state. */
        std::unique_ptr<Optim::Optimizer> optimizer;

        /** @brief The compiled step. */
        std::unique_ptr<Train::Trainer> trainer;
    };

    /** @brief Owns one run of a test. */
    using RunPointer = std::unique_ptr<TrainedRun>;

    /** @brief Makes the optimizer of @p kind over @p slots. */
    Result<std::unique_ptr<Optim::Optimizer>> make_optimizer(
        SushiAI::ExecutionBackend& backend, std::vector<Optim::ParameterSlot> slots, Kind kind,
        double momentum)
    {
        if (kind == Kind::SGD)
        {
            Optim::SGDOptions options;
            options.learning_rate = 0.1;
            options.momentum = momentum;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> made,
                          Optim::SGD::make(backend, std::move(slots), options));
            return std::unique_ptr<Optim::Optimizer>(std::move(made));
        }

        Optim::AdamWOptions options;
        options.learning_rate = 0.01;
        options.weight_decay = 0.01;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> made,
                      Optim::AdamW::make(backend, std::move(slots), options));
        return std::unique_ptr<Optim::Optimizer>(std::move(made));
    }

    /**
     * @brief Builds a run whose parameters are initialised from @p seed.
     * @return The run, or the first failure of a trace, a bind, an optimizer or the step.
     */
    Result<std::unique_ptr<TrainedRun>> make_run(SushiAI::ExecutionBackend& backend, Kind kind,
                                          std::uint64_t seed, double momentum = K_MOMENTUM)
    {
        std::unique_ptr<TrainedRun> run = std::make_unique<TrainedRun>(backend.engine());

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(run->model, K_BATCH, K_FEATURES, K_CLASSES));
        run->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), run->model));
        run->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, *run->built, *run->state, run->pool));

        NN::InitPolicy policy;
        policy.seed = seed;
        SA_TRY(NN::initialize(backend, *run->state, policy));

        SA_TRY_ASSIGN(run->optimizer, make_optimizer(backend, std::move(slots), kind, momentum));
        run->objective.emplace(SaTest::classifier_objective(*run->built));
        SA_TRY_ASSIGN(run->trainer,
                      Train::Trainer::make(backend, *run->built, run->pool, *run->objective,
                                           *run->optimizer));
        return std::move(run);
    }

    /** @brief Returns a rate that halves at the first update after the save. */
    Optim::StepSchedule make_schedule()
    {
        Optim::StepSchedule schedule;
        schedule.base_rate = 0.05;
        schedule.steps = {static_cast<std::int64_t>(K_HALF_UPDATES) + 1};
        schedule.scales = {0.5};
        return schedule;
    }

    /** @brief Runs @p count scheduled updates, the first of them update @p first of the run. */
    Status advance(TrainedRun& run, const Data::Dataset& dataset, std::uint64_t first,
                   std::uint64_t count)
    {
        const Optim::StepSchedule schedule = make_schedule();
        Data::Batch batch;
        for (std::uint64_t update = first; update < first + count; ++update)
        {
            const std::size_t index =
                static_cast<std::size_t>((update - 1) % static_cast<std::uint64_t>(K_BATCHES));
            SA_TRY(dataset.read(index, batch));

            (void)Optim::apply_step_schedule(schedule, *run.optimizer);
            SA_TRY(run.trainer->run_step(batch));
        }

        return Status();
    }

    /** @brief Returns every element of every entry of @p state as bit patterns. */
    std::vector<std::uint32_t> bits_of(NN::ModelState& state)
    {
        std::vector<std::uint32_t> bits;
        for (const NN::StateEntry& entry : state.entries())
        {
            Result<SushiBLAS::Tensor*> found = state.tensor(entry.state);
            SA_EXPECT_OK(found);
            const float* values = found.value()->data_as<float>();
            for (int64_t i = 0; i < found.value()->num_elements; ++i)
            {
                std::uint32_t pattern = 0;
                std::memcpy(&pattern, &values[i], sizeof(pattern));
                bits.push_back(pattern);
            }
        }

        return bits;
    }

    /** @brief Returns every element of every entry of the state of @p run as bit patterns. */
    std::vector<std::uint32_t> bits_of(TrainedRun& run)
    {
        return bits_of(*run.state);
    }

    /** @brief Returns the batches every run of a test reads. */
    Result<std::unique_ptr<Data::SyntheticClassification>> make_dataset(
        SushiAI::ExecutionBackend& backend)
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        return Data::SyntheticClassification::make(backend, options);
    }

    /**
     * @brief Checks that a save after half the updates and a restore into a new run give
     *        the parameters of the run that was never interrupted, bit for bit.
     */
    void expect_resume_is_bit_identical(SushiAI::ExecutionBackend& backend, Kind kind,
                                        const char* name)
    {
        const SaTest::ScratchFile file(name);
        SA_ASSERT_OK_AND_ASSIGN(const auto dataset, make_dataset(backend));

        SA_ASSERT_OK_AND_ASSIGN(const RunPointer whole, make_run(backend, kind, 11));
        SA_EXPECT_OK(advance(*whole, *dataset, 1, 2 * K_HALF_UPDATES));
        const std::vector<std::uint32_t> expected = bits_of(*whole);

        SA_ASSERT_OK_AND_ASSIGN(const RunPointer first, make_run(backend, kind, 11));
        SA_EXPECT_OK(advance(*first, *dataset, 1, K_HALF_UPDATES));
        SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *first->state, *first->optimizer));

        // Another seed, so a restore that did nothing would be visible.
        SA_ASSERT_OK_AND_ASSIGN(const RunPointer resumed, make_run(backend, kind, 22));
        ASSERT_NE(bits_of(*resumed), bits_of(*first));
        SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *resumed->state, *resumed->optimizer));
        EXPECT_EQ(resumed->optimizer->step_count(), K_HALF_UPDATES);
        EXPECT_EQ(bits_of(*resumed), bits_of(*first));

        SA_EXPECT_OK(advance(*resumed, *dataset, K_HALF_UPDATES + 1, K_HALF_UPDATES));
        EXPECT_EQ(resumed->optimizer->step_count(), 2 * K_HALF_UPDATES);
        EXPECT_EQ(bits_of(*resumed), expected);

        // The same file as weights alone restarts the schedule and the moments.
        SA_ASSERT_OK_AND_ASSIGN(const RunPointer restarted, make_run(backend, kind, 22));
        SA_EXPECT_OK(IO::load_checkpoint(file.path(), *restarted->state));
        EXPECT_EQ(restarted->optimizer->step_count(), 0u);
        EXPECT_EQ(bits_of(*restarted), bits_of(*first));
        SA_EXPECT_OK(advance(*restarted, *dataset, K_HALF_UPDATES + 1, K_HALF_UPDATES));
        EXPECT_NE(bits_of(*restarted), expected);
    }

    /** @brief Returns every byte of the file at @p path. */
    std::string read_bytes(const std::filesystem::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    /** @brief Replaces the file at @p path with @p bytes. */
    void write_bytes(const std::filesystem::path& path, const std::string& bytes)
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    /** @brief Names a classifier with one dropout between its two linear layers. */
    using DropoutMLP = NN::Sequential<NN::Linear, NN::ReLU, NN::Dropout, NN::Linear>;

    /** @brief Holds the seed the training graph of every dropout run is traced with. */
    constexpr std::uint64_t K_GRAPH_SEED = 20261011ull;

    /** @brief Holds the seed of the batch order of every epoch of the shuffled dataset. */
    constexpr std::uint64_t K_ORDER_SEED = 0x0DDBA11ull;

    /** @brief Holds the drop rate of the dropout runs. */
    constexpr double K_DROP_RATE = 0.25;

    /** @brief Holds the seed the parameters of the saved and the whole run start from. */
    constexpr std::uint64_t K_INIT_SEED = 11;

    /** @brief Holds the seed of a run whose parameters a load has to replace. */
    constexpr std::uint64_t K_OTHER_INIT_SEED = 22;

    /** @brief Counts the bytes of a one-element Int64 record beside its name. */
    constexpr std::size_t K_SCALAR_RECORD_BYTES = 28;

    /**
     * @brief Returns a dataset that reads the batches of @p inner in an order drawn anew at
     *        each epoch.
     * @param inner The dataset the batches come from; it must outlive the result.
     */
    std::unique_ptr<Data::InMemoryEpochDataset> shuffled(const Data::Dataset& inner)
    {
        const std::size_t count = inner.batch_count();
        return std::make_unique<Data::InMemoryEpochDataset>(
            count, inner.batch_size(), Data::describe_input(inner), inner.targets(),
            [&inner, count](std::int64_t epoch, std::size_t index)
            {
                const Data::EpochPermutation order(K_ORDER_SEED, epoch, count);
                Data::Batch batch;
                SA_EXPECT_OK(inner.read(static_cast<std::size_t>(order.at(index)), batch));
                return batch;
            });
    }

    /**
     * @struct DropoutRun
     * @brief Holds a model with dropout, its state, an optimizer and the trainer over them.
     */
    struct DropoutRun
    {
        /** @brief Builds the pool on @p engine; make_dropout_run() fills the rest. */
        explicit DropoutRun(SushiBLAS::Engine& engine) : pool(engine)
        {

        }

        /** @brief The model the graph is traced from. */
        DropoutMLP model = DropoutMLP(NN::Linear{K_FEATURES, 12}, NN::ReLU{},
                                      NN::Dropout{K_DROP_RATE}, NN::Linear{12, K_CLASSES});

        /** @brief The objective the graph is traced with. */
        Train::CrossEntropyObjective objective{K_BATCH, K_CLASSES, DType::Float32};

        /** @brief The training graph of the model. */
        std::optional<Train::TrainingGraph> built;

        /** @brief The parameters of the model. */
        std::optional<NN::ModelState> state;

        /** @brief The pool the graph's tensors live in. */
        Graph::TensorPool pool;

        /** @brief The optimizer over the parameters of the state. */
        std::unique_ptr<Optim::Optimizer> optimizer;

        /** @brief The trainer that owns the compiled step. */
        std::unique_ptr<Train::Trainer> trainer;
    };

    /** @brief Owns one dropout run of a test. */
    using DropoutRunPointer = std::unique_ptr<DropoutRun>;

    /**
     * @brief Builds a run with dropout, traced with K_GRAPH_SEED, under SGD with momentum.
     * @param seed Seeds the initial parameters.
     * @return The run, or the first failure of a trace, a bind, the optimizer or the step.
     */
    Result<DropoutRunPointer> make_dropout_run(SushiAI::ExecutionBackend& backend,
                                               std::uint64_t seed)
    {
        DropoutRunPointer run = std::make_unique<DropoutRun>(backend.engine());

        Train::GraphOptions options;
        options.seed = K_GRAPH_SEED;
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(run->model, run->objective,
                                                  Shape{K_BATCH, K_FEATURES}, options));
        run->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(backend.engine(), run->model));
        run->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(backend, *run->built, *run->state, run->pool));

        NN::InitPolicy policy;
        policy.seed = seed;
        SA_TRY(NN::initialize(backend, *run->state, policy));

        SA_TRY_ASSIGN(run->optimizer,
                      make_optimizer(backend, std::move(slots), Kind::SGD, K_MOMENTUM));
        SA_TRY_ASSIGN(run->trainer,
                      Train::Trainer::make(backend, *run->built, run->pool, run->objective,
                                           *run->optimizer));
        return std::move(run);
    }

    /** @brief Returns the progress of a run under K_GRAPH_SEED that has not trained yet. */
    IO::RunProgress start_of_run()
    {
        IO::RunProgress progress;
        progress.seed = K_GRAPH_SEED;
        return progress;
    }

    /** @brief Runs @p count updates from @p progress on and moves @p progress past them. */
    Status advance_from(DropoutRun& run, const Data::Dataset& dataset,
                        IO::RunProgress& progress, std::uint64_t count)
    {
        Data::Batch batch;
        for (std::uint64_t update = 0; update < count; ++update)
        {
            Data::begin_epoch(dataset, static_cast<std::int64_t>(progress.epoch));
            SA_TRY(dataset.read(static_cast<std::size_t>(progress.batch), batch));
            SA_TRY(run.trainer->run_step(batch));

            ++progress.batch;
            if (progress.batch == dataset.batch_count())
            {
                progress.batch = 0;
                ++progress.epoch;
            }
            progress.step_runs = run.trainer->step().run_count();
        }

        return Status();
    }

    /**
     * @brief Trains one run for 2 * K_HALF_UPDATES updates and another for half of them,
     *        which it saves to @p path with its progress.
     * @return The parameters of the run that was never interrupted, as bit patterns.
     */
    Result<std::vector<std::uint32_t>> train_whole_and_save_half(
        SushiAI::ExecutionBackend& backend, const Data::Dataset& dataset,
        const std::filesystem::path& path)
    {
        SA_TRY_ASSIGN(const DropoutRunPointer whole, make_dropout_run(backend, K_INIT_SEED));
        IO::RunProgress whole_progress = start_of_run();
        SA_TRY(advance_from(*whole, dataset, whole_progress, 2 * K_HALF_UPDATES));

        SA_TRY_ASSIGN(const DropoutRunPointer first, make_dropout_run(backend, K_INIT_SEED));
        IO::RunProgress first_progress = start_of_run();
        SA_TRY(advance_from(*first, dataset, first_progress, K_HALF_UPDATES));
        SA_TRY(IO::save_run_checkpoint(path, *first->state, *first->optimizer,
                                       first_progress));

        return bits_of(*whole->state);
    }

    /** @brief Returns how many bytes the four progress records take at the end of a file. */
    std::size_t progress_bytes()
    {
        return 4 * K_SCALAR_RECORD_BYTES + IO::K_RUN_SEED_STATE.size()
             + IO::K_RUN_STEP_RUNS_STATE.size() + IO::K_RUN_EPOCH_STATE.size()
             + IO::K_RUN_BATCH_STATE.size();
    }
} // namespace

namespace
{
    /** @brief Holds an optimizer kind and the scratch file its resume writes. */
    struct KindRow
    {
        std::string name;     /**< The row's name. */
        Kind kind;            /**< The optimizer the run trains with. */
        std::string file;     /**< The scratch file name. */
    };

    /** @brief Runs the resume promise over each optimizer kind. */
    class Medium_RunCheckpointResume : public SaTest::EngineTest,
                                       public ::testing::WithParamInterface<KindRow>
    {
    };
} // namespace

/** @brief Tests the resume across a change of the scheduled rate, once for each optimizer. */
TEST_P(Medium_RunCheckpointResume, AResumedRunMatchesTheUninterruptedOneBitForBit)
{
    expect_resume_is_bit_identical(backend(), GetParam().kind, GetParam().file.c_str());
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_RunCheckpointResume,
    ::testing::Values(KindRow{"SGDWithMomentum", Kind::SGD, "sushiai_run_checkpoint_sgd.sushi"},
                      KindRow{"AdamW", Kind::AdamW, "sushiai_run_checkpoint_adamw.sushi"}),
    [](const ::testing::TestParamInfo<KindRow>& info) { return info.param.name; });

/** @brief Tests that both savers, called right after a step, write that step's weights. */
TEST_F(Medium_RunCheckpoint, RegressionASaveRightAfterAStepWritesThatStepsWeights)
{
    const SaTest::ScratchFile weights_file("sushiai_save_after_step.sushi");
    const SaTest::ScratchFile run_file("sushiai_run_save_after_step.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const auto dataset, make_dataset(backend()));

    // A twin run settled on the host before it is read gives the step's weights.
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer twin, make_run(backend(), Kind::SGD, 11));
    const std::vector<std::uint32_t> before = bits_of(*twin);
    SA_EXPECT_OK(advance(*twin, *dataset, 1, 1));
    flush();
    const std::vector<std::uint32_t> stepped = bits_of(*twin);
    ASSERT_NE(stepped, before);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(advance(*saved, *dataset, 1, 1));
    SA_EXPECT_OK(IO::save_checkpoint(weights_file.path(), *saved->state));
    SA_EXPECT_OK(IO::save_run_checkpoint(run_file.path(), *saved->state, *saved->optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer from_weights, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_checkpoint(weights_file.path(), *from_weights->state));
    EXPECT_EQ(bits_of(*from_weights), stepped);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer from_run, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_run_checkpoint(run_file.path(), *from_run->state,
                                         *from_run->optimizer));
    EXPECT_EQ(bits_of(*from_run), stepped);
}

/** @brief Tests that a file with model state alone refuses a resume and says so. */
TEST_F(Medium_RunCheckpoint, AFileWithoutRunStateRefusesAResume)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_weights.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), *saved->state));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_weights.sushi",
                      "version 2", "no run state", "'weights'");
}

/** @brief Tests that a file with model state alone loads as weights. */
TEST_F(Medium_RunCheckpoint, AFileWithoutRunStateLoadsAsWeights)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_weights_load.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), *saved->state));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), *fresh->state));
    EXPECT_EQ(bits_of(*fresh), bits_of(*saved));
}

/** @brief Tests that a version 1 file refuses a resume and says so. */
TEST_F(Medium_RunCheckpoint, AVersionOneFileRefusesAResume)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_version_one.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), *saved->state));

    // Version 1 had no run-state count: the header was magic, version and tensor count.
    std::string bytes = read_bytes(file.path());
    ASSERT_GT(bytes.size(), K_OFFSET_RUN_STATE_COUNT + 4);
    bytes.erase(K_OFFSET_RUN_STATE_COUNT, 4);
    bytes[4] = 1;
    write_bytes(file.path(), bytes);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "version 1", "no run state");
}

/** @brief Tests that run state another optimizer type wrote is refused with both types. */
TEST_F(Medium_RunCheckpoint, RunStateOfAnotherOptimizerTypeIsRefused)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_other_type.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *saved->state, *saved->optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer other,
                            make_run(backend(), Kind::AdamW, 11));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *other->state, *other->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_other_type.sushi",
                      "'sgd'", "'adamw'");
}

namespace
{
    /** @brief Holds the momentum a file was written with, the one it loads into, and the words. */
    struct StateRow
    {
        std::string name;          /**< The row's name. */
        double written;            /**< The momentum of the run that wrote the file. */
        double loaded;             /**< The momentum of the run that loads it. */
        std::string first;         /**< A text the message holds. */
        std::string second;        /**< Another text the message holds. */
        std::string third;         /**< A third text the message holds. */
    };

    /** @brief Runs the run-state match over a missing and a surplus record. */
    class Medium_RunCheckpointState : public SaTest::EngineTest,
                                      public ::testing::WithParamInterface<StateRow>
    {
    };
} // namespace

/** @brief Tests that a run-state record the file lacks or holds too many of is refused by name. */
TEST_P(Medium_RunCheckpointState, ARunStateRecordTheOptimizerDisagreesOnIsRefusedByName)
{
    const StateRow& row = GetParam();
    const SaTest::ScratchFile file("sushiai_run_checkpoint_state.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer writer,
                            make_run(backend(), Kind::SGD, 11, row.written));
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer reader,
                            make_run(backend(), Kind::SGD, 11, row.loaded));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *writer->state, *writer->optimizer));

    SA_EXPECT_FAILURE(
        IO::load_run_checkpoint(file.path(), *reader->state, *reader->optimizer),
        StatusCode::Incompatible, row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_RunCheckpointState,
    ::testing::Values(
        StateRow{"Missing", 0.0, K_MOMENTUM, "does not supply", ".momentum'", "'sgd'"},
        StateRow{"Surplus", K_MOMENTUM, 0.0, ".momentum'", "does not keep", "'sgd'"}),
    [](const ::testing::TestParamInfo<StateRow>& info) { return info.param.name; });

namespace
{
    /** @brief Holds the progress a save is given, if any, and the progress a load then reads. */
    struct ProgressRow
    {
        std::string name;                          /**< The row's name. */
        std::optional<IO::RunProgress> given;      /**< The progress the save states, if any. */
        IO::RunProgress expected;                  /**< The progress the load reads. */
    };

    /** @brief Returns a progress holding the four fields given. */
    IO::RunProgress progress_of(std::uint64_t seed, std::uint64_t step_runs, std::uint64_t epoch,
                                std::uint64_t batch)
    {
        IO::RunProgress progress;
        progress.seed = seed;
        progress.step_runs = step_runs;
        progress.epoch = epoch;
        progress.batch = batch;
        return progress;
    }

    /** @brief Runs the progress round trip over a stated progress and an absent one. */
    class Medium_RunCheckpointProgress : public SaTest::EngineTest,
                                         public ::testing::WithParamInterface<ProgressRow>
    {
    };
} // namespace

/** @brief Tests that the progress of a run is written and read back, field for field. */
TEST_P(Medium_RunCheckpointProgress, ARunCheckpointHoldsTheSeedAndThePositionOfItsRun)
{
    const ProgressRow& row = GetParam();
    const SaTest::ScratchFile file("sushiai_run_checkpoint_progress.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer run, make_run(backend(), Kind::SGD, 11));

    // A save that states no progress writes the start of a run under the seed 0.
    SA_EXPECT_OK(row.given
                     ? IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer,
                                               *row.given)
                     : IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer));

    IO::RunProgress loaded;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer, &loaded));
    EXPECT_EQ(loaded.seed, row.expected.seed);
    EXPECT_EQ(loaded.step_runs, row.expected.step_runs);
    EXPECT_EQ(loaded.epoch, row.expected.epoch);
    EXPECT_EQ(loaded.batch, row.expected.batch);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_RunCheckpointProgress,
    ::testing::Values(ProgressRow{"Stated", progress_of(9, 6, 2, 3), progress_of(9, 6, 2, 3)},
                      ProgressRow{"Absent", std::nullopt, progress_of(0, 0, 0, 0)}),
    [](const ::testing::TestParamInfo<ProgressRow>& info) { return info.param.name; });

/** @brief Tests a resume in the middle of an epoch, with dropout and a shuffled order. */
TEST_F(Medium_RunCheckpoint, AResumedRunWithDropoutAndAShuffledOrderMatchesBitForBit)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_dropout.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const auto batches, make_dataset(backend()));
    const std::unique_ptr<Data::InMemoryEpochDataset> dataset = shuffled(*batches);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint32_t> expected,
                            train_whole_and_save_half(backend(), *dataset, file.path()));

    SA_ASSERT_OK_AND_ASSIGN(const DropoutRunPointer resumed,
                            make_dropout_run(backend(), K_OTHER_INIT_SEED));
    IO::RunProgress progress;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *resumed->state, *resumed->optimizer,
                                         &progress));

    // Six updates over epochs of four batches stop at the third batch of the second epoch.
    EXPECT_EQ(progress.seed, K_GRAPH_SEED);
    EXPECT_EQ(progress.step_runs, K_HALF_UPDATES);
    EXPECT_EQ(progress.epoch, 1u);
    EXPECT_EQ(progress.batch, 2u);

    SA_EXPECT_OK(resumed->trainer->restore_run_count(
        static_cast<std::size_t>(progress.step_runs)));
    SA_EXPECT_OK(advance_from(*resumed, *dataset, progress, K_HALF_UPDATES));

    EXPECT_EQ(progress.step_runs, 2 * K_HALF_UPDATES);
    EXPECT_EQ(progress.epoch, 3u);
    EXPECT_EQ(progress.batch, 0u);
    EXPECT_EQ(resumed->optimizer->step_count(), 2 * K_HALF_UPDATES);
    EXPECT_EQ(bits_of(*resumed->state), expected);
}


namespace
{
    /** @brief Writes a run checkpoint of @p run to @p path, then cuts the progress records off. */
    void save_without_progress(const std::filesystem::path& path, TrainedRun& run)
    {
        SA_EXPECT_OK(IO::save_run_checkpoint(path, *run.state, *run.optimizer));

        // The progress records close the file, so a file without them is a prefix and a count.
        std::string bytes = read_bytes(path);
        ASSERT_GT(bytes.size(), K_OFFSET_RUN_STATE_COUNT + 4 + progress_bytes());
        ASSERT_GT(bytes[K_OFFSET_RUN_STATE_COUNT], 4);
        bytes.resize(bytes.size() - progress_bytes());
        bytes[K_OFFSET_RUN_STATE_COUNT] = static_cast<char>(bytes[K_OFFSET_RUN_STATE_COUNT] - 4);
        write_bytes(path, bytes);
    }
} // namespace

/** @brief Tests that run state without the progress records is refused on a resume. */
TEST_F(Medium_RunCheckpoint, RunStateWithoutTheProgressRecordsIsRefusedOnResume)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_no_progress.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    save_without_progress(file.path(), *saved);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_no_progress.sushi",
                      "does not supply", "'run.seed'", "'weights'");
}

/** @brief Tests that run state without the progress records still loads as weights. */
TEST_F(Medium_RunCheckpoint, RunStateWithoutTheProgressRecordsStillLoadsAsWeights)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_no_progress_weights.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(backend(), Kind::SGD, 11));
    save_without_progress(file.path(), *saved);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(backend(), Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), *fresh->state));
    EXPECT_EQ(bits_of(*fresh), bits_of(*saved));
}

namespace
{
    /** @brief Holds a patch of the bytes of a run checkpoint and the refusal it earns. */
    struct BadProgressRow
    {
        std::string name;                                 /**< The row's name. */
        std::function<void(std::string&)> patch;          /**< Changes the written bytes. */
        StatusCode code;                                  /**< The code of the refusal. */
        std::string first;                                /**< A text the message holds. */
        std::string second;                               /**< Another text the message holds. */
    };

    /** @brief Runs the progress refusal over a record of another type and one out of range. */
    class Medium_RunCheckpointBadProgress : public SaTest::EngineTest,
                                            public ::testing::WithParamInterface<BadProgressRow>
    {
    };
} // namespace

/** @brief Tests that a progress record of another type, or an epoch too large, is refused. */
TEST_P(Medium_RunCheckpointBadProgress, AProgressRecordOfAnotherTypeOrRangeIsRefused)
{
    const BadProgressRow& row = GetParam();
    const SaTest::ScratchFile file("sushiai_run_checkpoint_bad_progress.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer run, make_run(backend(), Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer));
    std::string bytes = read_bytes(file.path());
    ASSERT_GT(bytes.size(), progress_bytes());
    row.patch(bytes);
    write_bytes(file.path(), bytes);

    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer),
                      row.code, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Medium_RunCheckpointBadProgress,
    ::testing::Values(
        // The last record ends with its type, its rank, one extent and eight payload bytes.
        BadProgressRow{"AnotherType",
                       [](std::string& bytes)
                       { bytes[bytes.size() - 24] = static_cast<char>(DType::Float64); },
                       StatusCode::Incompatible, "'run.batch'", "progress of a run"},
        // The payload of 'run.epoch' ends where the record of 'run.batch' starts.
        BadProgressRow{"AnEpochTooLarge",
                       [](std::string& bytes)
                       {
                           const std::size_t batch_record =
                               K_SCALAR_RECORD_BYTES + IO::K_RUN_BATCH_STATE.size();
                           bytes[bytes.size() - batch_record - 1] = static_cast<char>(0x80);
                       },
                       StatusCode::InvalidFile, "'run.epoch'", "largest epoch index"}),
    [](const ::testing::TestParamInfo<BadProgressRow>& info) { return info.param.name; });
