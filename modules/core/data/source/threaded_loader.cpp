/**************************************************************************/
/* threaded_loader.cpp                                                    */
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
 * @file threaded_loader.cpp
 * @brief Implements the dataset that reads the batches of another one ahead, on threads.
 * @author Mustafa Garip
 */

#include <SushiAI/data/threaded_loader.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Data
    {
        namespace
        {
            /** @brief Names this unit in a message. */
            constexpr std::string_view K_LOADER = "ThreadedLoader";

            /** @brief Names where the batch of one slot stands. */
            enum class SlotState : std::uint8_t
            {
                Idle,    /**< Holds no index; no thread reads it. */
                Queued,  /**< Holds an index no thread has taken yet. */
                Running, /**< A thread is reading its index into its batch. */
                Ready,   /**< Its batch and its status wait for the caller. */
            };

            /** @brief Holds one batch that is read ahead, and the status of its read. */
            struct Slot
            {
                SlotState state = SlotState::Idle; /**< Where the batch stands. */
                std::size_t index = 0;             /**< Which batch it is; unset when Idle. */
                Status status;                     /**< What the read returned, once Ready. */
                Batch batch;                       /**< Written by one thread while Running. */
            };
        } // namespace

        /** @brief Holds what the caller's thread and the reading threads share. */
        struct ThreadedLoader::State
        {
            /** @brief Takes the dataset and makes one idle slot per batch read ahead. */
            State(std::unique_ptr<Dataset> dataset, std::size_t prefetch)
                : inner(std::move(dataset)), slots(prefetch)
            {

            }

            State(const State&) = delete;
            State& operator=(const State&) = delete;
            State(State&&) = delete;
            State& operator=(State&&) = delete;

            /** @brief Stops the threads and joins them. */
            ~State()
            {
                stop();
            }

            /** @brief Returns the queued slot of the lowest index, or null; mutex held. */
            [[nodiscard]] Slot* first_queued() noexcept
            {
                Slot* found = nullptr;
                for (Slot& slot : slots)
                {
                    if (slot.state == SlotState::Queued
                        && (found == nullptr || slot.index < found->index))
                        found = &slot;
                }
                return found;
            }

            /** @brief Returns the slot that holds @p index, or null; mutex held. */
            [[nodiscard]] Slot* slot_of(std::size_t index) noexcept
            {
                for (Slot& slot : slots)
                {
                    if (slot.state != SlotState::Idle && slot.index == index)
                        return &slot;
                }
                return nullptr;
            }

            /** @brief Reports whether a thread is reading into a slot; mutex held. */
            [[nodiscard]] bool is_reading() const noexcept
            {
                for (const Slot& slot : slots)
                {
                    if (slot.state == SlotState::Running)
                        return true;
                }
                return false;
            }

            /** @brief Reads queued slots until stop(); the body of every thread. */
            void run_worker() noexcept
            {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;)
                {
                    Slot* slot = nullptr;
                    work.wait(lock, [&]()
                    {
                        slot = first_queued();
                        return is_stopping || slot != nullptr;
                    });
                    if (is_stopping)
                        return;

                    slot->state = SlotState::Running;
                    const std::size_t index = slot->index;
                    lock.unlock();

                    // A Running slot belongs to this thread alone until it is Ready.
                    Status status = Detail::guard(Detail::SeamPhase::Execute, [&]()
                    {
                        return inner->read(index, slot->batch);
                    });

                    lock.lock();
                    slot->status = std::move(status);
                    slot->state = SlotState::Ready;
                    progress.notify_all();
                }
            }

            /** @brief Drops every batch read ahead and waits for the running reads. */
            void quiesce(std::unique_lock<std::mutex>& lock)
            {
                for (Slot& slot : slots)
                {
                    if (slot.state != SlotState::Running)
                        slot.state = SlotState::Idle;
                }

                progress.wait(lock, [&]() { return !is_reading(); });

                for (Slot& slot : slots)
                {
                    slot.state = SlotState::Idle;
                    slot.status = Status();
                }
            }

            /** @brief Queues the batches from @p index on, one per slot; every slot is idle. */
            void schedule_from(std::size_t index)
            {
                const std::size_t count = inner->batch_count();
                next_read = index;
                next_queued = index;

                for (Slot& slot : slots)
                {
                    if (next_queued >= count)
                        break;

                    slot.index = next_queued++;
                    slot.state = SlotState::Queued;
                }

                is_started = true;
                work.notify_all();
            }

            /** @brief Waits for batch @p index, exchanges it with @p batch and queues one more. */
            [[nodiscard]] Status take(std::size_t index, Batch& batch)
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!is_started || index != next_read)
                {
                    quiesce(lock);
                    schedule_from(index);
                }

                Slot* const slot = slot_of(index);
                SA_FATAL_IF(slot == nullptr, "{}: no slot holds batch {}", K_LOADER, index);
                progress.wait(lock, [&]() { return slot->state == SlotState::Ready; });

                const Status status = slot->status;
                std::swap(batch, slot->batch);
                slot->status = Status();
                slot->state = SlotState::Idle;
                next_read = index + 1;

                if (next_queued < inner->batch_count())
                {
                    slot->index = next_queued++;
                    slot->state = SlotState::Queued;
                    work.notify_one();
                }

                return status;
            }

            /** @brief Forwards an epoch with no read running, then reads ahead from 0. */
            [[nodiscard]] Status enter_epoch(std::int64_t epoch)
            {
                std::unique_lock<std::mutex> lock(mutex);
                quiesce(lock);
                Data::begin_epoch(*inner, epoch);
                schedule_from(0);
                return Status();
            }

            /** @brief Tells every thread to return and joins it; running reads finish first. */
            void stop() noexcept
            {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    is_stopping = true;
                }
                work.notify_all();

                for (std::thread& thread : threads)
                {
                    if (thread.joinable())
                        thread.join();
                }
            }

            /** @brief Holds the dataset the threads read. */
            std::unique_ptr<Dataset> inner;

            /** @brief Guards every member below it. */
            std::mutex mutex;

            /** @brief Wakes a thread when a slot is queued or the loader stops. */
            std::condition_variable work;

            /** @brief Wakes the caller when a read has finished. */
            std::condition_variable progress;

            /** @brief Holds one slot per batch read ahead; never resized. */
            std::vector<Slot> slots;

            /** @brief Holds the index the caller reads next when it reads in order. */
            std::size_t next_read = 0;

            /** @brief Holds the first index no slot was given. */
            std::size_t next_queued = 0;

            /** @brief Tells whether the slots were queued since the last epoch began. */
            bool is_started = false;

            /** @brief Tells the threads to return. */
            bool is_stopping = false;

            /** @brief Holds the reading threads, joined by stop(). */
            std::vector<std::thread> threads;
        };

        Result<std::unique_ptr<ThreadedLoader>> ThreadedLoader::make(
            std::unique_ptr<Dataset> inner, const LoaderOptions& options)
        {
            SA_FAIL_IF(inner == nullptr, StatusCode::InvalidDescription,
                       "{}: there is no dataset to read", K_LOADER);
            SA_FAIL_IF(options.workers < 1 || options.prefetch < 1,
                       StatusCode::InvalidDescription,
                       "{}: '{}' is read by {} threads, {} batches ahead; both must be at "
                       "least 1",
                       K_LOADER, inner->name(), options.workers, options.prefetch);

            return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
            {
                std::unique_ptr<State> state = std::make_unique<State>(
                    std::move(inner), static_cast<std::size_t>(options.prefetch));
                State* const shared = state.get();

                // A thread that fails to start leaves the State to join the ones before it.
                state->threads.reserve(static_cast<std::size_t>(options.workers));
                for (std::int32_t worker = 0; worker < options.workers; ++worker)
                    state->threads.emplace_back([shared]() { shared->run_worker(); });

                return std::unique_ptr<ThreadedLoader>(new ThreadedLoader(std::move(state)));
            });
        }

        ThreadedLoader::ThreadedLoader(std::unique_ptr<State> state) noexcept
            : state_(std::move(state))
        {

        }

        ThreadedLoader::~ThreadedLoader() = default;

        int64_t ThreadedLoader::batch_size() const noexcept
        {
            return state_->inner->batch_size();
        }

        Shape ThreadedLoader::sample_shape() const
        {
            return state_->inner->sample_shape();
        }

        DType ThreadedLoader::sample_dtype() const noexcept
        {
            return state_->inner->sample_dtype();
        }

        const std::vector<TargetDescription>& ThreadedLoader::targets() const noexcept
        {
            return state_->inner->targets();
        }

        std::size_t ThreadedLoader::batch_count() const noexcept
        {
            return state_->inner->batch_count();
        }

        std::string_view ThreadedLoader::name() const noexcept
        {
            return state_->inner->name();
        }

        Status ThreadedLoader::read(std::size_t index, Batch& batch) const
        {
            SA_FATAL_IF(index >= batch_count(),
                        "{}: '{}' has {} batches; batch {} is outside them", K_LOADER, name(),
                        batch_count(), index);

            return Detail::guard(Detail::SeamPhase::Execute, [&]()
            {
                return state_->take(index, batch);
            });
        }

        void ThreadedLoader::begin_epoch(std::int64_t epoch) const noexcept
        {
            SA_FATAL_IF(epoch < 0, "{}: '{}' was given the epoch {}; epochs count from 0",
                        K_LOADER, name(), epoch);

            const Status entered = Detail::guard(Detail::SeamPhase::Execute, [&]()
            {
                return state_->enter_epoch(epoch);
            });
            SA_FATAL_IF(!entered.is_ok(), "{}: '{}' could not enter epoch {}: {}", K_LOADER,
                        name(), epoch, entered.message());
        }

    } // namespace Data
} // namespace SushiAI
