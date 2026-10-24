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
 * @file test_run_checkpoint.cpp
 * @brief Tests that a run resumed from a checkpoint with run state continues bit for bit.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/classifier_objective/classifier_objective.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/scratch_files/scratch_file.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>

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
        SushiBLAS::Engine& engine, std::vector<Optim::ParameterSlot> slots, Kind kind,
        double momentum)
    {
        if (kind == Kind::SGD)
        {
            Optim::SGDOptions options;
            options.learning_rate = 0.1;
            options.momentum = momentum;
            SA_TRY_ASSIGN(std::unique_ptr<Optim::SGD> made,
                          Optim::SGD::make(engine, std::move(slots), options));
            return std::unique_ptr<Optim::Optimizer>(std::move(made));
        }

        Optim::AdamWOptions options;
        options.learning_rate = 0.01;
        options.weight_decay = 0.01;
        SA_TRY_ASSIGN(std::unique_ptr<Optim::AdamW> made,
                      Optim::AdamW::make(engine, std::move(slots), options));
        return std::unique_ptr<Optim::Optimizer>(std::move(made));
    }

    /**
     * @brief Builds a run whose parameters are initialised from @p seed.
     * @return The run, or the first failure of a trace, a bind, an optimizer or the step.
     */
    Result<std::unique_ptr<TrainedRun>> make_run(SushiBLAS::Engine& engine, Kind kind,
                                          std::uint64_t seed, double momentum = K_MOMENTUM)
    {
        std::unique_ptr<TrainedRun> run = std::make_unique<TrainedRun>(engine);

        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_classifier(run->model, K_BATCH, K_FEATURES, K_CLASSES));
        run->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, run->model));
        run->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(*run->built, *run->state, run->pool));

        NN::InitPolicy policy;
        policy.seed = seed;
        SA_TRY(NN::initialize(engine, *run->state, policy));

        SA_TRY_ASSIGN(run->optimizer, make_optimizer(engine, std::move(slots), kind, momentum));
        run->objective.emplace(SaTest::classifier_objective(*run->built));
        SA_TRY_ASSIGN(run->trainer,
                      Train::Trainer::make(engine, *run->built, run->pool, *run->objective,
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
    Result<std::unique_ptr<Data::SyntheticClassification>> make_dataset(SushiBLAS::Engine& engine)
    {
        Data::SyntheticOptions options;
        options.batch_size = K_BATCH;
        options.batches = K_BATCHES;
        options.features = K_FEATURES;
        options.classes = K_CLASSES;
        return Data::SyntheticClassification::make(engine, options);
    }

    /**
     * @brief Checks that a save after half the updates and a restore into a new run give
     *        the parameters of the run that was never interrupted, bit for bit.
     */
    void expect_resume_is_bit_identical(SushiBLAS::Engine& engine, Kind kind, const char* name)
    {
        const SaTest::ScratchFile file(name);
        SA_ASSERT_OK_AND_ASSIGN(const auto dataset, make_dataset(engine));

        SA_ASSERT_OK_AND_ASSIGN(const RunPointer whole, make_run(engine, kind, 11));
        SA_EXPECT_OK(advance(*whole, *dataset, 1, 2 * K_HALF_UPDATES));
        const std::vector<std::uint32_t> expected = bits_of(*whole);

        SA_ASSERT_OK_AND_ASSIGN(const RunPointer first, make_run(engine, kind, 11));
        SA_EXPECT_OK(advance(*first, *dataset, 1, K_HALF_UPDATES));
        SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *first->state, *first->optimizer));

        // Another seed, so a restore that did nothing would be visible.
        SA_ASSERT_OK_AND_ASSIGN(const RunPointer resumed, make_run(engine, kind, 22));
        ASSERT_NE(bits_of(*resumed), bits_of(*first));
        SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *resumed->state, *resumed->optimizer));
        EXPECT_EQ(resumed->optimizer->step_count(), K_HALF_UPDATES);
        EXPECT_EQ(bits_of(*resumed), bits_of(*first));

        SA_EXPECT_OK(advance(*resumed, *dataset, K_HALF_UPDATES + 1, K_HALF_UPDATES));
        EXPECT_EQ(resumed->optimizer->step_count(), 2 * K_HALF_UPDATES);
        EXPECT_EQ(bits_of(*resumed), expected);

        // The same file as weights alone restarts the schedule and the moments.
        SA_ASSERT_OK_AND_ASSIGN(const RunPointer restarted, make_run(engine, kind, 22));
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

    /** @brief Reads the batches of another dataset in an order drawn anew at each epoch. */
    class ShuffledBatches final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /** @brief Reads through @p inner, which must outlive this. */
            explicit ShuffledBatches(const Data::Dataset& inner) : inner_(&inner)
            {

            }

            int64_t batch_size() const noexcept override { return inner_->batch_size(); }
            Shape sample_shape() const override { return inner_->sample_shape(); }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return inner_->targets();
            }
            std::size_t batch_count() const noexcept override { return inner_->batch_count(); }
            std::string_view name() const noexcept override { return "shuffled"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override
            {
                const Data::EpochPermutation order(K_ORDER_SEED, epoch_, inner_->batch_count());
                return inner_->read(static_cast<std::size_t>(order.at(index)), batch);
            }

            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override
            {
                const Data::EpochPermutation order(K_ORDER_SEED, epoch_, inner_->batch_count());
                return inner_->read_rows(static_cast<std::size_t>(order.at(index)), rows, batch);
            }

            void begin_epoch(std::int64_t epoch) const noexcept override
            {
                epoch_ = epoch;
            }

        private:
            /** @brief Points at the dataset the batches come from. */
            const Data::Dataset* inner_;

            /** @brief Holds the epoch the next read draws its order for. */
            mutable std::int64_t epoch_ = 0;
    };

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
    Result<DropoutRunPointer> make_dropout_run(SushiBLAS::Engine& engine, std::uint64_t seed)
    {
        DropoutRunPointer run = std::make_unique<DropoutRun>(engine);

        Train::GraphOptions options;
        options.seed = K_GRAPH_SEED;
        SA_TRY_ASSIGN(Train::TrainingGraph built,
                      Train::build_training_graph(run->model, run->objective,
                                                  Shape{K_BATCH, K_FEATURES}, options));
        run->built.emplace(std::move(built));

        SA_TRY_ASSIGN(NN::ModelState state, NN::ModelState::create(engine, run->model));
        run->state.emplace(std::move(state));

        SA_TRY_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                      Train::bind_parameters(*run->built, *run->state, run->pool));

        NN::InitPolicy policy;
        policy.seed = seed;
        SA_TRY(NN::initialize(engine, *run->state, policy));

        SA_TRY_ASSIGN(run->optimizer,
                      make_optimizer(engine, std::move(slots), Kind::SGD, K_MOMENTUM));
        SA_TRY_ASSIGN(run->trainer,
                      Train::Trainer::make(engine, *run->built, run->pool, run->objective,
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
        SushiBLAS::Engine& engine, const Data::Dataset& dataset,
        const std::filesystem::path& path)
    {
        SA_TRY_ASSIGN(const DropoutRunPointer whole, make_dropout_run(engine, K_INIT_SEED));
        IO::RunProgress whole_progress = start_of_run();
        SA_TRY(advance_from(*whole, dataset, whole_progress, 2 * K_HALF_UPDATES));

        SA_TRY_ASSIGN(const DropoutRunPointer first, make_dropout_run(engine, K_INIT_SEED));
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

/** @brief Tests the resume of SGD with momentum across a change of the scheduled rate. */
TEST_F(Medium_RunCheckpoint, AResumedSGDRunMatchesTheUninterruptedOneBitForBit)
{
    expect_resume_is_bit_identical(*engine, Kind::SGD, "sushiai_run_checkpoint_sgd.sushi");
}

/** @brief Tests the resume of AdamW across a change of the scheduled rate. */
TEST_F(Medium_RunCheckpoint, AResumedAdamWRunMatchesTheUninterruptedOneBitForBit)
{
    expect_resume_is_bit_identical(*engine, Kind::AdamW, "sushiai_run_checkpoint_adamw.sushi");
}

/** @brief Tests that both savers, called right after a step, write that step's weights. */
TEST_F(Medium_RunCheckpoint, RegressionASaveRightAfterAStepWritesThatStepsWeights)
{
    const SaTest::ScratchFile weights_file("sushiai_save_after_step.sushi");
    const SaTest::ScratchFile run_file("sushiai_run_save_after_step.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const auto dataset, make_dataset(*engine));

    // A twin run settled on the host before it is read gives the step's weights.
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer twin, make_run(*engine, Kind::SGD, 11));
    const std::vector<std::uint32_t> before = bits_of(*twin);
    SA_EXPECT_OK(advance(*twin, *dataset, 1, 1));
    flush();
    const std::vector<std::uint32_t> stepped = bits_of(*twin);
    ASSERT_NE(stepped, before);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(advance(*saved, *dataset, 1, 1));
    SA_EXPECT_OK(IO::save_checkpoint(weights_file.path(), *saved->state));
    SA_EXPECT_OK(IO::save_run_checkpoint(run_file.path(), *saved->state, *saved->optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer from_weights, make_run(*engine, Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_checkpoint(weights_file.path(), *from_weights->state));
    EXPECT_EQ(bits_of(*from_weights), stepped);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer from_run, make_run(*engine, Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_run_checkpoint(run_file.path(), *from_run->state,
                                         *from_run->optimizer));
    EXPECT_EQ(bits_of(*from_run), stepped);
}

/** @brief Tests that a file with model state alone loads as weights and refuses a resume. */
TEST_F(Medium_RunCheckpoint, AFileWithoutRunStateIsWeightsAndNoResume)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_weights.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), *saved->state));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(*engine, Kind::SGD, 22));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_weights.sushi",
                      "version 2", "no run state", "'weights'");
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), *fresh->state));
    EXPECT_EQ(bits_of(*fresh), bits_of(*saved));
}

/** @brief Tests that a version 1 file still loads as weights and refuses a resume. */
TEST_F(Medium_RunCheckpoint, AVersionOneFileStillLoadsAsWeights)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_version_one.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_checkpoint(file.path(), *saved->state));

    // Version 1 had no run-state count: the header was magic, version and tensor count.
    std::string bytes = read_bytes(file.path());
    ASSERT_GT(bytes.size(), K_OFFSET_RUN_STATE_COUNT + 4);
    bytes.erase(K_OFFSET_RUN_STATE_COUNT, 4);
    bytes[4] = 1;
    write_bytes(file.path(), bytes);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(*engine, Kind::SGD, 22));
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), *fresh->state));
    EXPECT_EQ(bits_of(*fresh), bits_of(*saved));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "version 1", "no run state");
}

/** @brief Tests that a version above the newest is refused with the versions read. */
TEST_F(Medium_RunCheckpoint, AFutureVersionIsRefusedByBothLoaders)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_future.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer run, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer));

    std::string bytes = read_bytes(file.path());
    ASSERT_GT(bytes.size(), 8u);
    bytes[4] = 3;
    write_bytes(file.path(), bytes);

    SA_EXPECT_FAILURE(IO::load_checkpoint(file.path(), *run->state), StatusCode::InvalidFile,
                      "version 3", "versions 1 to 2");
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer),
                      StatusCode::InvalidFile, "version 3", "versions 1 to 2");
}

/** @brief Tests that run state another optimizer type wrote is refused with both types. */
TEST_F(Medium_RunCheckpoint, RunStateOfAnotherOptimizerTypeIsRefused)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_other_type.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *saved->state, *saved->optimizer));

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer other,
                            make_run(*engine, Kind::AdamW, 11));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *other->state, *other->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_other_type.sushi",
                      "'sgd'", "'adamw'");
}

/** @brief Tests that a missing and a surplus run-state record are each refused by name. */
TEST_F(Medium_RunCheckpoint, AMissingOrSurplusRunStateRecordIsRefusedByName)
{
    const SaTest::ScratchFile plain_file("sushiai_run_checkpoint_plain.sushi");
    const SaTest::ScratchFile momentum_file("sushiai_run_checkpoint_momentum.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer plain,
                            make_run(*engine, Kind::SGD, 11, 0.0));
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer with_momentum,
                            make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(plain_file.path(), *plain->state, *plain->optimizer));
    SA_EXPECT_OK(IO::save_run_checkpoint(momentum_file.path(), *with_momentum->state,
                                         *with_momentum->optimizer));

    SA_EXPECT_FAILURE(IO::load_run_checkpoint(plain_file.path(), *with_momentum->state,
                                              *with_momentum->optimizer),
                      StatusCode::Incompatible, "does not supply", ".momentum'",
                      "'sgd'");
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(momentum_file.path(), *plain->state,
                                              *plain->optimizer),
                      StatusCode::Incompatible, ".momentum'", "does not keep", "'sgd'");
}

/** @brief Tests that the progress of a run is written and read back, field for field. */
TEST_F(Medium_RunCheckpoint, ARunCheckpointHoldsTheSeedAndThePositionOfItsRun)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_progress.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const RunPointer run, make_run(*engine, Kind::SGD, 11));

    IO::RunProgress saved;
    saved.seed = 9;
    saved.step_runs = 6;
    saved.epoch = 2;
    saved.batch = 3;
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer,
                                         saved));

    IO::RunProgress loaded;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer, &loaded));
    EXPECT_EQ(loaded.seed, 9u);
    EXPECT_EQ(loaded.step_runs, 6u);
    EXPECT_EQ(loaded.epoch, 2u);
    EXPECT_EQ(loaded.batch, 3u);

    SA_EXPECT_OK(IO::check_run_seed(loaded, 9, file.path().string()));
    SA_EXPECT_FAILURE(IO::check_run_seed(loaded, 10, file.path().string()),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_progress.sushi",
                      "seed 9", "seed 10");

    // A save that states no progress writes the start of a run under the seed 0.
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer));
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer, &loaded));
    EXPECT_EQ(loaded.seed, 0u);
    EXPECT_EQ(loaded.step_runs, 0u);
    EXPECT_EQ(loaded.epoch, 0u);
    EXPECT_EQ(loaded.batch, 0u);
}

/** @brief Tests a resume in the middle of an epoch, with dropout and a shuffled order. */
TEST_F(Medium_RunCheckpoint, AResumedRunWithDropoutAndAShuffledOrderMatchesBitForBit)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_dropout.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const auto batches, make_dataset(*engine));
    const ShuffledBatches dataset(*batches);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint32_t> expected,
                            train_whole_and_save_half(*engine, dataset, file.path()));

    SA_ASSERT_OK_AND_ASSIGN(const DropoutRunPointer resumed,
                            make_dropout_run(*engine, K_OTHER_INIT_SEED));
    IO::RunProgress progress;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *resumed->state, *resumed->optimizer,
                                         &progress));

    // Six updates over epochs of four batches stop at the third batch of the second epoch.
    EXPECT_EQ(progress.seed, K_GRAPH_SEED);
    EXPECT_EQ(progress.step_runs, K_HALF_UPDATES);
    EXPECT_EQ(progress.epoch, 1u);
    EXPECT_EQ(progress.batch, 2u);

    SA_EXPECT_OK(IO::check_run_seed(progress, K_GRAPH_SEED, file.path().string()));
    SA_EXPECT_OK(resumed->trainer->restore_run_count(
        static_cast<std::size_t>(progress.step_runs)));
    SA_EXPECT_OK(advance_from(*resumed, dataset, progress, K_HALF_UPDATES));

    EXPECT_EQ(progress.step_runs, 2 * K_HALF_UPDATES);
    EXPECT_EQ(progress.epoch, 3u);
    EXPECT_EQ(progress.batch, 0u);
    EXPECT_EQ(resumed->optimizer->step_count(), 2 * K_HALF_UPDATES);
    EXPECT_EQ(bits_of(*resumed->state), expected);
}

/** @brief Tests that a resume without the run count, or without the position, diverges. */
TEST_F(Medium_RunCheckpoint, AResumeThatDropsTheRunCountOrThePositionDiverges)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_dropped.sushi");
    SA_ASSERT_OK_AND_ASSIGN(const auto batches, make_dataset(*engine));
    const ShuffledBatches dataset(*batches);

    SA_ASSERT_OK_AND_ASSIGN(const std::vector<std::uint32_t> expected,
                            train_whole_and_save_half(*engine, dataset, file.path()));

    // The masks of runs 0 to 5 are drawn again where the saved run would draw 6 to 11.
    SA_ASSERT_OK_AND_ASSIGN(const DropoutRunPointer no_count,
                            make_dropout_run(*engine, K_OTHER_INIT_SEED));
    IO::RunProgress kept;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *no_count->state, *no_count->optimizer,
                                         &kept));
    SA_EXPECT_OK(advance_from(*no_count, dataset, kept, K_HALF_UPDATES));
    EXPECT_NE(bits_of(*no_count->state), expected);

    // The order of epoch 0 is read again from its first batch.
    SA_ASSERT_OK_AND_ASSIGN(const DropoutRunPointer no_position,
                            make_dropout_run(*engine, K_OTHER_INIT_SEED));
    IO::RunProgress moved;
    SA_EXPECT_OK(IO::load_run_checkpoint(file.path(), *no_position->state,
                                         *no_position->optimizer, &moved));
    SA_EXPECT_OK(no_position->trainer->restore_run_count(
        static_cast<std::size_t>(moved.step_runs)));
    moved.epoch = 0;
    moved.batch = 0;
    SA_EXPECT_OK(advance_from(*no_position, dataset, moved, K_HALF_UPDATES));
    EXPECT_NE(bits_of(*no_position->state), expected);
}

/** @brief Tests that run state without the progress records is refused and still is weights. */
TEST_F(Medium_RunCheckpoint, RunStateWithoutTheProgressRecordsIsRefusedOnResume)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_no_progress.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer saved, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *saved->state, *saved->optimizer));

    // The progress records close the file, so a file without them is a prefix and a count.
    std::string bytes = read_bytes(file.path());
    ASSERT_GT(bytes.size(), K_OFFSET_RUN_STATE_COUNT + 4 + progress_bytes());
    ASSERT_GT(bytes[K_OFFSET_RUN_STATE_COUNT], 4);
    bytes.resize(bytes.size() - progress_bytes());
    bytes[K_OFFSET_RUN_STATE_COUNT] = static_cast<char>(bytes[K_OFFSET_RUN_STATE_COUNT] - 4);
    write_bytes(file.path(), bytes);

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer fresh, make_run(*engine, Kind::SGD, 22));
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *fresh->state, *fresh->optimizer),
                      StatusCode::Incompatible, "sushiai_run_checkpoint_no_progress.sushi",
                      "does not supply", "'run.seed'", "'weights'");
    SA_EXPECT_OK(IO::load_checkpoint(file.path(), *fresh->state));
    EXPECT_EQ(bits_of(*fresh), bits_of(*saved));
}

/** @brief Tests that a progress record of another type, or an epoch too large, is refused. */
TEST_F(Medium_RunCheckpoint, AProgressRecordOfAnotherTypeOrRangeIsRefused)
{
    const SaTest::ScratchFile file("sushiai_run_checkpoint_bad_progress.sushi");

    SA_ASSERT_OK_AND_ASSIGN(const RunPointer run, make_run(*engine, Kind::SGD, 11));
    SA_EXPECT_OK(IO::save_run_checkpoint(file.path(), *run->state, *run->optimizer));
    const std::string written = read_bytes(file.path());
    ASSERT_GT(written.size(), progress_bytes());

    // The last record ends with its type, its rank, one extent and eight payload bytes.
    std::string retyped = written;
    retyped[retyped.size() - 24] = static_cast<char>(DType::Float64);
    write_bytes(file.path(), retyped);
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer),
                      StatusCode::Incompatible, "'run.batch'", "progress of a run");

    // The payload of 'run.epoch' ends where the record of 'run.batch' starts.
    std::string huge = written;
    const std::size_t batch_record = K_SCALAR_RECORD_BYTES + IO::K_RUN_BATCH_STATE.size();
    huge[huge.size() - batch_record - 1] = static_cast<char>(0x80);
    write_bytes(file.path(), huge);
    SA_EXPECT_FAILURE(IO::load_run_checkpoint(file.path(), *run->state, *run->optimizer),
                      StatusCode::InvalidFile, "'run.epoch'", "largest epoch index");
}
