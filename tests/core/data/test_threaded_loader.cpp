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
 * @file test_threaded_loader.cpp
 * @brief Tests the threaded loader: the order, the bound on reads ahead, a failed read, the epoch
 *        and the shutdown, over a dataset that needs no device.
 * @author Mustafa Garip
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

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

        /** @brief Counts the reads that have started and not yet returned. */
        std::atomic<std::int64_t> in_flight{0};

        /** @brief Holds how many reads were running when the dataset was destroyed. */
        std::atomic<std::int64_t> in_flight_at_destruction{-1};
    };

    /** @brief A dataset whose batch names its epoch and its index and holds no tensor. */
    class RecordingDataset final : public Data::Dataset, public Data::EpochAware
    {
        public:
            /** @brief Records into @p probe; the read of @p failing returns InvalidFile. */
            RecordingDataset(Probe& probe, std::size_t failing) : probe_(&probe), failing_(failing)
            {

            }

            ~RecordingDataset() override
            {
                probe_->in_flight_at_destruction.store(probe_->in_flight.load());
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
                const InFlight running(*probe_);
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

            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange,
                                           Data::Batch& batch) const override
            {
                return read(index, batch);
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
            /** @brief Counts one read in flight for as long as it lives. */
            class InFlight
            {
                public:
                    /** @brief Counts a read of @p probe. */
                    explicit InFlight(Probe& probe) : probe_(probe)
                    {
                        probe_.in_flight.fetch_add(1);
                    }

                    InFlight(const InFlight&) = delete;
                    InFlight& operator=(const InFlight&) = delete;

                    ~InFlight()
                    {
                        probe_.in_flight.fetch_sub(1);
                    }

                private:
                    /** @brief Receives the count. */
                    Probe& probe_;
            };

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

    /** @brief Returns the indices from @p first up to, not including, @p last. */
    std::vector<std::size_t> indices_from(std::size_t first, std::size_t last)
    {
        std::vector<std::size_t> indices(last - first);
        std::iota(indices.begin(), indices.end(), first);
        return indices;
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

    /** @brief Checks that reading each of @p indices in turn gives the batch of that index and @p epoch. */
    void expect_reads_named(const Data::Dataset& loader, Probe& probe, std::int64_t epoch,
                            const std::vector<std::size_t>& indices)
    {
        Data::Batch batch;
        for (std::size_t index : indices)
            EXPECT_EQ(read_name(loader, probe, index, batch),
                      RecordingDataset::name_of(epoch, index));
    }

    /** @brief Checks that two loaders give the same batches for each of @p epochs, in order. */
    void expect_loaders_agree(const Data::Dataset& first, Probe& first_probe,
                              const Data::Dataset& second, Probe& second_probe,
                              std::int64_t epochs)
    {
        Data::Batch first_batch;
        Data::Batch second_batch;
        for (std::int64_t epoch = 0; epoch < epochs; ++epoch)
        {
            first_probe.asked.store(-1);
            second_probe.asked.store(-1);
            Data::begin_epoch(first, epoch);
            Data::begin_epoch(second, epoch);

            for (std::size_t index = 0; index < K_BATCHES; ++index)
            {
                const std::string expected = RecordingDataset::name_of(epoch, index);
                EXPECT_EQ(read_name(first, first_probe, index, first_batch), expected);
                EXPECT_EQ(read_name(second, second_probe, index, second_batch), expected);
            }
        }
    }
} // namespace

/** @brief Tests that reading an index hands over the batch of that index. */
TEST(Medium_ThreadedLoader, TheBatchOfEachIndexIsHandedOverAsThatIndex)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 3, 4);
    ASSERT_NE(loader, nullptr);

    expect_reads_named(*loader, probe, 0, indices_from(0, K_BATCHES));
}

/** @brief Tests that the batches are read on at most the worker threads and not on the caller. */
TEST(Medium_ThreadedLoader, TheBatchesAreReadOnWorkerThreadsOnly)
{
    constexpr std::int32_t K_WORKERS = 3;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, K_WORKERS, 4);
    ASSERT_NE(loader, nullptr);

    expect_reads_named(*loader, probe, 0, indices_from(0, K_BATCHES));

    const std::lock_guard<std::mutex> lock(probe.mutex);
    EXPECT_FALSE(probe.threads.empty());
    EXPECT_LE(probe.threads.size(), static_cast<std::size_t>(K_WORKERS));
    EXPECT_EQ(probe.threads.count(std::this_thread::get_id()), 0u);
}

/** @brief Tests that no read starts further ahead of the caller than the prefetch count. */
TEST(Medium_ThreadedLoader, ReadsNoFurtherAheadThanThePrefetchCount)
{
    constexpr std::int32_t K_PREFETCH = 3;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 4, K_PREFETCH);
    ASSERT_NE(loader, nullptr);

    expect_reads_named(*loader, probe, 0, indices_from(0, K_BATCHES));

    EXPECT_LE(probe.furthest_ahead.load(), K_PREFETCH);
}

/** @brief Tests that a failed read is the status of its index, with the failure named. */
TEST(Medium_ThreadedLoader, AFailedReadIsReportedWithTheStatusOfItsIndex)
{
    constexpr std::size_t K_FAILING = 5;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4, K_FAILING);
    ASSERT_NE(loader, nullptr);
    expect_reads_named(*loader, probe, 0, indices_from(0, K_FAILING));
    Data::Batch batch;
    probe.asked.store(static_cast<std::int64_t>(K_FAILING));

    SA_EXPECT_FAILURE(loader->read(K_FAILING, batch), StatusCode::InvalidFile, "batch 5",
                      "unreadable");
}

/** @brief Tests that the indices after a failed read are read as if it had not failed. */
TEST(Medium_ThreadedLoader, ReadsAfterAFailedReadStillSucceed)
{
    constexpr std::size_t K_FAILING = 5;

    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4, K_FAILING);
    ASSERT_NE(loader, nullptr);
    expect_reads_named(*loader, probe, 0, indices_from(0, K_FAILING));
    Data::Batch batch;
    probe.asked.store(static_cast<std::int64_t>(K_FAILING));
    EXPECT_FALSE(loader->read(K_FAILING, batch).is_ok());

    expect_reads_named(*loader, probe, 0, indices_from(K_FAILING + 1, K_BATCHES));
}

/** @brief Tests that an index out of order is read, and the indices after it in order. */
TEST(Medium_ThreadedLoader, AnIndexOutOfOrderIsRead)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 2);
    ASSERT_NE(loader, nullptr);

    expect_reads_named(*loader, probe, 0, {7, 8, 2, 3, 11, 0});
}

/** @brief Tests that begin_epoch hands the epoch to the dataset. */
TEST(Medium_ThreadedLoader, BeginEpochReachesTheDataset)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4);
    ASSERT_NE(loader, nullptr);

    Data::begin_epoch(*loader, 3);

    EXPECT_EQ(probe.epoch.load(), 3);
}

/** @brief Tests that the reads after begin_epoch are the batches of the new epoch. */
TEST(Medium_ThreadedLoader, BeginEpochDropsWhatWasReadAhead)
{
    Probe probe;
    const std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 2, 4);
    ASSERT_NE(loader, nullptr);
    expect_reads_named(*loader, probe, 0, {0, 1});
    probe.asked.store(-1);

    Data::begin_epoch(*loader, 3);

    expect_reads_named(*loader, probe, 3, indices_from(0, K_BATCHES));
}

/** @brief Tests that two loaders of different thread and read-ahead counts deliver alike. */
TEST(Medium_ThreadedLoader, TwoLoadersOfDifferentWidthDeliverTheSameEpochsInTheSameOrder)
{
    constexpr std::int64_t K_EPOCHS = 2;

    Probe narrow_probe;
    Probe wide_probe;
    const std::unique_ptr<Data::ThreadedLoader> narrow = make_loader(narrow_probe, 1, 1);
    const std::unique_ptr<Data::ThreadedLoader> wide = make_loader(wide_probe, 4, 6);
    ASSERT_NE(narrow, nullptr);
    ASSERT_NE(wide, nullptr);

    expect_loaders_agree(*narrow, narrow_probe, *wide, wide_probe, K_EPOCHS);
}

/** @brief Tests that the destructor waits for the reads that run and destroys the dataset after. */
TEST(Medium_ThreadedLoader, TheDestructorJoinsWithBatchesReadAhead)
{
    Probe probe;
    std::unique_ptr<Data::ThreadedLoader> loader = make_loader(probe, 3, 4);
    ASSERT_NE(loader, nullptr);
    expect_reads_named(*loader, probe, 0, {0});

    loader.reset();

    EXPECT_EQ(probe.in_flight_at_destruction.load(), 0);
}

/** @brief Tests that make refuses a null dataset. */
TEST(Medium_ThreadedLoader, MakeRefusesNoDataset)
{
    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(nullptr, Data::LoaderOptions{}),
                      StatusCode::InvalidDescription, "no dataset");
}

/** @brief Tests that make refuses a worker count below 1 and names it. */
TEST(Medium_ThreadedLoader, MakeRefusesNoWorker)
{
    Probe probe;
    Data::LoaderOptions options;
    options.workers = 0;

    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(
                          std::make_unique<RecordingDataset>(probe, K_NO_FAILURE), options),
                      StatusCode::InvalidDescription, "0 threads");
}

/** @brief Tests that make refuses a prefetch count below 1 and names it. */
TEST(Medium_ThreadedLoader, MakeRefusesNoPrefetch)
{
    Probe probe;
    Data::LoaderOptions options;
    options.prefetch = 0;

    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(
                          std::make_unique<RecordingDataset>(probe, K_NO_FAILURE), options),
                      StatusCode::InvalidDescription, "0 batches ahead");
}

namespace
{
    /** @brief Provides the engine the inner dataset of a loader over generated data is made on. */
    class Medium_ThreadedLoaderRows : public SushiAI::Test::EngineTest
    {

    };
} // namespace

/** @brief Tests that the loader states the extents, the type and the name its inner dataset states. */
TEST_F(Medium_ThreadedLoaderRows, TheLoaderStatesWhatItsInnerDatasetStates)
{
    Data::RingsOptions options;
    options.batch_size = 4;
    options.batches = 3;
    SA_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Data::Rings> rings,
                            Data::Rings::make(*engine, options));
    const Data::Rings* const inner = rings.get();

    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::ThreadedLoader> loader,
                            Data::ThreadedLoader::make(std::move(rings), Data::LoaderOptions{}));

    EXPECT_EQ(loader->batch_size(), inner->batch_size());
    EXPECT_EQ(loader->batch_count(), inner->batch_count());
    EXPECT_EQ(loader->name(), inner->name());
    EXPECT_EQ(loader->sample_dtype(), inner->sample_dtype());
    EXPECT_EQ(loader->sample_shape(), inner->sample_shape());
    EXPECT_EQ(loader->targets().size(), inner->targets().size());
}
