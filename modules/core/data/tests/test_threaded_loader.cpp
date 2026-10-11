/**************************************************************************/
/* test_threaded_loader.cpp                                               */
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
 * @file test_threaded_loader.cpp
 * @brief Tests the threaded loader: the order, the bound on reads ahead, a failed read, the
 *        epoch and the shutdown, over a dataset that needs no device.
 * @author Mustafa Garip
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;

namespace
{
    /** @brief The batch count of every dataset below. */
    constexpr std::size_t K_BATCHES = 12;

    /** @brief An index no batch has, for a dataset whose every read succeeds. */
    constexpr std::size_t K_NO_FAILURE = K_BATCHES;

    /** @brief Holds what a RecordingDataset saw, for the test that owns it. */
    struct Probe
    {
        /** @brief Guards threads. */
        std::mutex mutex;

        /** @brief Holds every thread that called read(). */
        std::set<std::thread::id> threads;

        /** @brief Holds the highest index the test has asked the loader for; -1 for none. */
        std::atomic<std::int64_t> asked{-1};

        /** @brief Holds how far past asked the furthest read() started. */
        std::atomic<std::int64_t> furthest_ahead{0};

        /** @brief Holds the epoch the last begin_epoch() gave. */
        std::atomic<std::int64_t> epoch{0};
    };

    /** @brief A dataset whose batch names its epoch and its index and holds no tensor. */
    class RecordingDataset final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /** @brief Records into @p probe; the read of @p failing returns InvalidFile. */
            RecordingDataset(Probe& probe, std::size_t failing) : probe_(&probe), failing_(failing)
            {

            }

            int64_t batch_size() const noexcept override { return 1; }
            Shape sample_shape() const override { return Shape{1}; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return K_BATCHES; }
            std::string_view name() const noexcept override { return "recording"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override
            {
                {
                    const std::lock_guard<std::mutex> lock(probe_->mutex);
                    probe_->threads.insert(std::this_thread::get_id());
                }

                const std::int64_t ahead = static_cast<std::int64_t>(index) - probe_->asked.load();
                std::int64_t seen = probe_->furthest_ahead.load();
                while (ahead > seen && !probe_->furthest_ahead.compare_exchange_weak(seen, ahead))
                    continue;

                SA_FAIL_IF(index == failing_, StatusCode::InvalidFile, "batch {} is unreadable",
                           index);

                batch.targets.assign(1, Data::BatchTarget{name_of(probe_->epoch.load(), index),
                                                          SushiBLAS::Tensor()});
                return Status();
            }

            void begin_epoch(std::int64_t epoch) const noexcept override
            {
                probe_->epoch.store(epoch);
            }

            /** @brief Returns "<epoch>:<index>", the role of the one target of a batch. */
            static std::string name_of(std::int64_t epoch, std::size_t index)
            {
                return std::to_string(epoch) + ":" + std::to_string(index);
            }

        private:
            /** @brief Receives what read() and begin_epoch() saw. */
            Probe* probe_;

            /** @brief Holds the index whose read fails. */
            std::size_t failing_;

            /** @brief Holds no description; the batches carry a name alone. */
            std::vector<Data::TargetDescription> targets_;
    };

    /** @brief Returns a loader over a RecordingDataset. */
    std::unique_ptr<Data::ThreadedLoader> make_loader(Probe& probe, std::int32_t workers,
                                                      std::int32_t prefetch,
                                                      std::size_t failing = K_NO_FAILURE)
    {
        Data::LoaderOptions options;
        options.workers = workers;
        options.prefetch = prefetch;

        auto made = Data::ThreadedLoader::make(
            std::make_unique<RecordingDataset>(probe, failing), options);
        EXPECT_TRUE(made.has_value()) << made.status().message();
        return made.has_value() ? std::move(made).value() : nullptr;
    }

    /** @brief Reads one batch through the loader and returns the name it carries. */
    std::string read_name(const Data::Dataset& loader, Probe& probe, std::size_t index,
                          Data::Batch& batch)
    {
        probe.asked.store(static_cast<std::int64_t>(index));

        const Status status = loader.read(index, batch);
        EXPECT_TRUE(status.is_ok()) << status.message();
        return status.is_ok() && batch.targets.size() == 1 ? batch.targets.front().role
                                                            : std::string();
    }
} // namespace

/** @brief Tests that the batches arrive in index order and were read on other threads. */
TEST(Unit_ThreadedLoader, HandsTheBatchesOverInIndexOrder)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 3, 4);
    ASSERT_NE(loader, nullptr);

    Data::Batch batch;
    for (std::size_t index = 0; index < K_BATCHES; ++index)
        EXPECT_EQ(read_name(*loader, probe, index, batch),
                  RecordingDataset::name_of(0, index));

    const std::lock_guard<std::mutex> lock(probe.mutex);
    EXPECT_FALSE(probe.threads.empty());
    EXPECT_LE(probe.threads.size(), 3u);
    EXPECT_EQ(probe.threads.count(std::this_thread::get_id()), 0u);
}

/** @brief Tests that no read starts further ahead of the caller than the prefetch count. */
TEST(Unit_ThreadedLoader, ReadsNoFurtherAheadThanThePrefetchCount)
{
    constexpr std::int32_t K_PREFETCH = 3;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 4, K_PREFETCH);
    ASSERT_NE(loader, nullptr);

    Data::Batch batch;
    for (std::size_t index = 0; index < K_BATCHES; ++index)
        read_name(*loader, probe, index, batch);

    EXPECT_LE(probe.furthest_ahead.load(), K_PREFETCH);
}

/** @brief Tests that a failed read is the status of its index and that later ones read. */
TEST(Unit_ThreadedLoader, AFailedReadIsTheStatusOfItsIndexAlone)
{
    constexpr std::size_t K_FAILING = 5;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4, K_FAILING);
    ASSERT_NE(loader, nullptr);

    Data::Batch batch;
    for (std::size_t index = 0; index < K_BATCHES; ++index)
    {
        if (index != K_FAILING)
        {
            EXPECT_EQ(read_name(*loader, probe, index, batch),
                      RecordingDataset::name_of(0, index));
            continue;
        }

        probe.asked.store(static_cast<std::int64_t>(index));
        SA_EXPECT_FAILURE(loader->read(index, batch), StatusCode::InvalidFile, "batch 5",
                          "unreadable");
    }
}

/** @brief Tests that an index out of order is read, and the indices after it in order. */
TEST(Unit_ThreadedLoader, AnIndexOutOfOrderIsRead)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 2);
    ASSERT_NE(loader, nullptr);

    Data::Batch batch;
    for (std::size_t index : {7u, 8u, 2u, 3u, 11u, 0u})
        EXPECT_EQ(read_name(*loader, probe, index, batch),
                  RecordingDataset::name_of(0, index));
}

/** @brief Tests that begin_epoch reaches the dataset and that the next reads are of it. */
TEST(Unit_ThreadedLoader, BeginEpochReachesTheDatasetAndDropsWhatWasReadAhead)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4);
    ASSERT_NE(loader, nullptr);

    Data::Batch batch;
    EXPECT_EQ(read_name(*loader, probe, 0, batch), RecordingDataset::name_of(0, 0));
    EXPECT_EQ(read_name(*loader, probe, 1, batch), RecordingDataset::name_of(0, 1));

    probe.asked.store(-1);
    Data::begin_epoch(*loader, 3);
    EXPECT_EQ(probe.epoch.load(), 3);

    for (std::size_t index = 0; index < K_BATCHES; ++index)
        EXPECT_EQ(read_name(*loader, probe, index, batch),
                  RecordingDataset::name_of(3, index));
}

/** @brief Tests that a loader with reads queued and running is destroyed without a read. */
TEST(Unit_ThreadedLoader, TheDestructorJoinsWithBatchesReadAhead)
{
    Probe probe;
    std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 3, 4);
    ASSERT_NE(loader, nullptr);

    Data::begin_epoch(*loader, 0);
    loader.reset();
    SUCCEED();
}

/** @brief Tests that the loader states the batch size, the count and the name of its dataset. */
TEST(Unit_ThreadedLoader, StatesWhatItsDatasetStates)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 1, 1);
    ASSERT_NE(loader, nullptr);

    EXPECT_EQ(loader->batch_size(), 1);
    EXPECT_EQ(loader->batch_count(), K_BATCHES);
    EXPECT_EQ(loader->name(), "recording");
    EXPECT_EQ(loader->sample_dtype(), DType::Float32);
    EXPECT_TRUE(loader->targets().empty());
}

/** @brief Tests that make refuses a null dataset and an option below 1, naming the option. */
TEST(Unit_ThreadedLoader, MakeRefusesNoDatasetAndAnOptionBelowOne)
{
    Probe probe;
    Data::LoaderOptions options;

    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(nullptr, options),
                      StatusCode::InvalidDescription, "no dataset");

    options.workers = 0;
    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(
                          std::make_unique<RecordingDataset>(probe, K_NO_FAILURE), options),
                      StatusCode::InvalidDescription, "0 threads");

    options.workers = 1;
    options.prefetch = 0;
    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(
                          std::make_unique<RecordingDataset>(probe, K_NO_FAILURE), options),
                      StatusCode::InvalidDescription, "0 batches ahead");
}

/** @brief Tests that begin_epoch leaves a dataset that is not EpochAware as it is. */
TEST(Unit_Epoch, ADatasetThatIsNotEpochAwareIsLeftAlone)
{
    class Plain final : public Data::Dataset
    {
        public:
            int64_t batch_size() const noexcept override { return 1; }
            Shape sample_shape() const override { return Shape{1}; }
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return 1; }
            std::string_view name() const noexcept override { return "plain"; }
            [[nodiscard]] Status read(std::size_t, Data::Batch&) const override
            {
                return Status();
            }

        private:
            /** @brief Holds no description. */
            std::vector<Data::TargetDescription> targets_;
    };

    const Plain plain;
    Data::begin_epoch(plain, 4);
    EXPECT_EQ(plain.batch_count(), 1u);
}
