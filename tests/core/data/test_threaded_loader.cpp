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


/**
 * @file test_threaded_loader.cpp
 * @brief Tests the threaded loader: the order, the bound on reads ahead, a failed read, the epoch
 *        and the shutdown, over datasets that need no device.
 * @author Mustafa Garip
 */

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>

using namespace SushiAI;

namespace
{
    /** @brief The batch count of every dataset below. */
    constexpr std::size_t K_BATCHES = 12;

    /** @brief The prefetch count of the loaders whose reads ahead are counted. */
    constexpr std::int32_t K_PREFETCH = 3;

    /** @brief Returns "<epoch>:<index>", the role of the one target of a batch. */
    std::string name_of(std::int64_t epoch, std::size_t index)
    {
        return std::to_string(epoch) + ":" + std::to_string(index);
    }

    /** @brief Holds what the threads that read a dataset left behind, for the test that owns it. */
    struct Probe
    {
        /** @brief Guards threads. */
        std::mutex mutex;

        /** @brief Holds every thread that built a batch. */
        std::set<std::thread::id> threads;

        /** @brief Counts the batches being built. */
        std::atomic<std::int64_t> in_flight{0};

        /** @brief Holds how long a batch takes to build; zero builds it at once. */
        std::chrono::milliseconds pause{1};
    };

    /** @brief Builds the batch of an epoch and an index: one target named by both, no tensor. */
    Data::Batch named_batch(std::int64_t epoch, std::size_t index)
    {
        Data::Batch batch;
        batch.targets.assign(1, Data::BatchTarget{name_of(epoch, index), SushiBLAS::Tensor()});
        return batch;
    }

    /** @brief Returns a dataset of K_BATCHES batches, each named by its epoch and index. */
    std::unique_ptr<Data::InMemoryEpochDataset> named_dataset(Probe* probe = nullptr)
    {
        return std::make_unique<Data::InMemoryEpochDataset>(
            K_BATCHES, 1, Data::InputDescription{Shape{1}, DType::Float32},
            std::vector<Data::TargetDescription>{},
            [probe](std::int64_t epoch, std::size_t index)
            {
                if (probe == nullptr)
                    return named_batch(epoch, index);

                probe->in_flight.fetch_add(1);
                {
                    const std::lock_guard<std::mutex> lock(probe->mutex);
                    probe->threads.insert(std::this_thread::get_id());
                }
                if (probe->pause.count() > 0)
                    std::this_thread::sleep_for(probe->pause);
                Data::Batch batch = named_batch(epoch, index);
                probe->in_flight.fetch_sub(1);
                return batch;
            });
    }

    /** @brief Holds a loader and the dataset it reads, which it owns. */
    struct Loading
    {
        /** @brief The dataset the loader reads; owned by @ref loader. */
        Data::InMemoryEpochDataset* dataset = nullptr;

        /** @brief The loader. */
        std::unique_ptr<Data::ThreadedLoader> loader;
    };

    /** @brief Returns a loader over @p dataset, which it takes. */
    Loading load(std::unique_ptr<Data::InMemoryEpochDataset> dataset, Data::LoaderOptions options)
    {
        Loading loading;
        loading.dataset = dataset.get();

        auto made = Data::ThreadedLoader::make(std::move(dataset), options);
        EXPECT_TRUE(made.has_value()) << made.status().message();
        if (made.has_value())
            loading.loader = std::move(made).value();
        return loading;
    }

    /** @brief Returns a loader that reads on the calling thread, @p prefetch batches ahead. */
    Loading load_inline(std::int32_t prefetch, std::size_t failing = K_BATCHES)
    {
        std::unique_ptr<Data::InMemoryEpochDataset> dataset = named_dataset();
        if (failing < K_BATCHES)
            dataset->fail_at(failing);

        Data::LoaderOptions options;
        options.prefetch = prefetch;
        options.executor = Data::ReadExecutor::CallingThread;
        return load(std::move(dataset), options);
    }

    /** @brief Returns the indices from @p first up to, not including, @p last. */
    std::vector<std::size_t> indices_from(std::size_t first, std::size_t last)
    {
        std::vector<std::size_t> indices(last - first);
        std::iota(indices.begin(), indices.end(), first);
        return indices;
    }

    /** @brief Reads one batch through the loader and returns the name it carries. */
    std::string read_name(const Data::Dataset& loader, std::size_t index, Data::Batch& batch)
    {
        const Status status = loader.read(index, batch);
        EXPECT_TRUE(status.is_ok()) << status.message();
        return status.is_ok() && batch.targets.size() == 1 ? batch.targets.front().role
                                                            : std::string();
    }

    /** @brief Checks that reading each of @p indices in turn gives the batch of that index and @p epoch. */
    void expect_reads_named(const Data::Dataset& loader, std::int64_t epoch,
                            const std::vector<std::size_t>& indices)
    {
        Data::Batch batch;
        for (std::size_t index : indices)
            EXPECT_EQ(read_name(loader, index, batch), name_of(epoch, index));
    }

    /** @brief Checks that two loaders give the same batches for each of @p epochs, in order. */
    void expect_loaders_agree(const Data::Dataset& first, const Data::Dataset& second,
                              std::int64_t epochs)
    {
        Data::Batch first_batch;
        Data::Batch second_batch;
        for (std::int64_t epoch = 0; epoch < epochs; ++epoch)
        {
            Data::begin_epoch(first, epoch);
            Data::begin_epoch(second, epoch);

            for (std::size_t index = 0; index < K_BATCHES; ++index)
            {
                const std::string expected = name_of(epoch, index);
                EXPECT_EQ(read_name(first, index, first_batch), expected);
                EXPECT_EQ(read_name(second, index, second_batch), expected);
            }
        }
    }

    /** @brief Runs each case with a sequence of indices read in that order. */
    class Small_ThreadedLoaderOrder : public ::testing::TestWithParam<std::vector<std::size_t>>
    {

    };
} // namespace

/** @brief Tests that reading an index hands over the batch of that index. */
TEST(Small_ThreadedLoader, TheBatchOfEachIndexIsHandedOverAsThatIndex)
{
    const Loading loading = load_inline(4);
    ASSERT_NE(loading.loader, nullptr);

    expect_reads_named(*loading.loader, 0, indices_from(0, K_BATCHES));
}

/** @brief Tests that a loader that reads on the calling thread starts no thread. */
TEST(Small_ThreadedLoader, ACallingThreadLoaderStartsNoThread)
{
    Probe probe;
    probe.pause = std::chrono::milliseconds{0};
    Data::LoaderOptions options;
    options.executor = Data::ReadExecutor::CallingThread;
    const Loading loading = load(named_dataset(&probe), options);
    ASSERT_NE(loading.loader, nullptr);

    expect_reads_named(*loading.loader, 0, indices_from(0, K_BATCHES));

    const std::lock_guard<std::mutex> lock(probe.mutex);
    EXPECT_EQ(probe.threads, std::set<std::thread::id>{std::this_thread::get_id()});
}

/** @brief Tests that no read starts further ahead of the caller than the prefetch count. */
TEST(Small_ThreadedLoader, ReadsNoFurtherAheadThanThePrefetchCount)
{
    const Loading loading = load_inline(K_PREFETCH);
    ASSERT_NE(loading.loader, nullptr);
    Data::Batch batch;

    for (std::size_t index = 0; index < K_BATCHES; ++index)
    {
        ASSERT_TRUE(loading.loader->read(index, batch).is_ok());

        const std::vector<std::size_t> journal = loading.dataset->journal();
        EXPECT_EQ(*std::max_element(journal.begin(), journal.end()),
                  std::min(index + K_PREFETCH, K_BATCHES - 1));
    }
}

/** @brief Tests that the batches are read in index order and each exactly once. */
TEST(Small_ThreadedLoader, EachBatchIsReadOnceInIndexOrder)
{
    const Loading loading = load_inline(K_PREFETCH);
    ASSERT_NE(loading.loader, nullptr);

    expect_reads_named(*loading.loader, 0, indices_from(0, K_BATCHES));

    EXPECT_EQ(loading.dataset->journal(), indices_from(0, K_BATCHES));
}

/** @brief Tests that a failed read is the status of its index, with the failure named. */
TEST(Small_ThreadedLoader, AFailedReadIsReportedWithTheStatusOfItsIndex)
{
    constexpr std::size_t K_FAILING = 5;

    const Loading loading = load_inline(4, K_FAILING);
    ASSERT_NE(loading.loader, nullptr);
    expect_reads_named(*loading.loader, 0, indices_from(0, K_FAILING));
    Data::Batch batch;

    SA_EXPECT_FAILURE(loading.loader->read(K_FAILING, batch), StatusCode::InvalidFile, "batch 5",
                      "unreadable");
}

/** @brief Tests that the indices after a failed read are read as if it had not failed. */
TEST(Small_ThreadedLoader, ReadsAfterAFailedReadStillSucceed)
{
    constexpr std::size_t K_FAILING = 5;

    const Loading loading = load_inline(4, K_FAILING);
    ASSERT_NE(loading.loader, nullptr);
    expect_reads_named(*loading.loader, 0, indices_from(0, K_FAILING));
    Data::Batch batch;
    EXPECT_FALSE(loading.loader->read(K_FAILING, batch).is_ok());

    expect_reads_named(*loading.loader, 0, indices_from(K_FAILING + 1, K_BATCHES));
}

/** @brief Tests that an index out of order is read, and the indices after it in order. */
TEST_P(Small_ThreadedLoaderOrder, AnIndexOutOfOrderIsRead)
{
    const Loading loading = load_inline(2);
    ASSERT_NE(loading.loader, nullptr);

    expect_reads_named(*loading.loader, 0, GetParam());
}

INSTANTIATE_TEST_SUITE_P(Orders, Small_ThreadedLoaderOrder,
                         ::testing::Values(std::vector<std::size_t>{7},
                                           std::vector<std::size_t>{7, 8, 2},
                                           std::vector<std::size_t>{7, 8, 2, 3, 11},
                                           std::vector<std::size_t>{11, 0}));

/** @brief Tests that begin_epoch hands the epoch to the dataset. */
TEST(Small_ThreadedLoader, BeginEpochReachesTheDataset)
{
    const Loading loading = load_inline(4);
    ASSERT_NE(loading.loader, nullptr);

    Data::begin_epoch(*loading.loader, 3);

    EXPECT_EQ(loading.dataset->epoch(), 3);
}

/** @brief Tests that the reads after begin_epoch are the batches of the new epoch. */
TEST(Small_ThreadedLoader, BeginEpochDropsWhatWasReadAhead)
{
    const Loading loading = load_inline(4);
    ASSERT_NE(loading.loader, nullptr);
    expect_reads_named(*loading.loader, 0, {0, 1});

    Data::begin_epoch(*loading.loader, 3);

    expect_reads_named(*loading.loader, 3, indices_from(0, K_BATCHES));
}

/** @brief Tests that two loaders of different read-ahead counts deliver alike. */
TEST(Small_ThreadedLoader, TwoLoadersOfDifferentPrefetchDeliverTheSameEpochsInTheSameOrder)
{
    constexpr std::int64_t K_EPOCHS = 2;

    const Loading narrow = load_inline(1);
    const Loading wide = load_inline(6);
    ASSERT_NE(narrow.loader, nullptr);
    ASSERT_NE(wide.loader, nullptr);

    expect_loaders_agree(*narrow.loader, *wide.loader, K_EPOCHS);
}

/** @brief Tests that make refuses a null dataset. */
TEST(Small_ThreadedLoader, MakeRefusesNoDataset)
{
    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(nullptr, Data::LoaderOptions{}),
                      StatusCode::InvalidDescription, "no dataset");
}

/** @brief Tests that make refuses a worker count below 1 and names it. */
TEST(Small_ThreadedLoader, MakeRefusesNoWorker)
{
    Data::LoaderOptions options;
    options.workers = 0;

    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(named_dataset(), options),
                      StatusCode::InvalidDescription, "0 threads");
}

/** @brief Tests that make refuses a prefetch count below 1 and names it. */
TEST(Small_ThreadedLoader, MakeRefusesNoPrefetch)
{
    Data::LoaderOptions options;
    options.prefetch = 0;

    SA_EXPECT_FAILURE(Data::ThreadedLoader::make(named_dataset(), options),
                      StatusCode::InvalidDescription, "0 batches ahead");
}

/** @brief Tests that the batches are read on at most the worker threads and not on the caller. */
TEST(Medium_ThreadedLoader, TheBatchesAreReadOnWorkerThreadsOnly)
{
    constexpr std::int32_t K_WORKERS = 3;

    Probe probe;
    Data::LoaderOptions options;
    options.workers = K_WORKERS;
    options.prefetch = 4;
    const Loading loading = load(named_dataset(&probe), options);
    ASSERT_NE(loading.loader, nullptr);

    expect_reads_named(*loading.loader, 0, indices_from(0, K_BATCHES));

    const std::lock_guard<std::mutex> lock(probe.mutex);
    EXPECT_FALSE(probe.threads.empty());
    EXPECT_LE(probe.threads.size(), static_cast<std::size_t>(K_WORKERS));
    EXPECT_EQ(probe.threads.count(std::this_thread::get_id()), 0u);
}

/** @brief Tests that two loaders of different thread and read-ahead counts deliver alike. */
TEST(Medium_ThreadedLoader, TwoLoadersOfDifferentWidthDeliverTheSameEpochsInTheSameOrder)
{
    constexpr std::int64_t K_EPOCHS = 2;

    Data::LoaderOptions narrow_options;
    narrow_options.workers = 1;
    narrow_options.prefetch = 1;
    Data::LoaderOptions wide_options;
    wide_options.workers = 4;
    wide_options.prefetch = 6;
    const Loading narrow = load(named_dataset(), narrow_options);
    const Loading wide = load(named_dataset(), wide_options);
    ASSERT_NE(narrow.loader, nullptr);
    ASSERT_NE(wide.loader, nullptr);

    expect_loaders_agree(*narrow.loader, *wide.loader, K_EPOCHS);
}

/** @brief Tests that the destructor waits for the reads that run before it returns. */
TEST(Medium_ThreadedLoader, TheDestructorJoinsWithBatchesReadAhead)
{
    Probe probe;
    Data::LoaderOptions options;
    options.workers = 3;
    options.prefetch = 4;
    Loading loading = load(named_dataset(&probe), options);
    ASSERT_NE(loading.loader, nullptr);
    expect_reads_named(*loading.loader, 0, {0});

    loading.loader.reset();

    EXPECT_EQ(probe.in_flight.load(), 0);
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
