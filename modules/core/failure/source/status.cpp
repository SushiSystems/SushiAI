/**************************************************************************/
/* status.cpp                                                             */
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
 * @file status.cpp
 * @brief Implements the shared failure block behind Status and the accessors that read it.
 * @author Mustafa Garip
 */

#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure/failure_block_counter.hpp>

#include <atomic>
#include <new>

namespace SushiAI
{
    namespace Detail
    {
        /** @brief Holds one failure's count, code and text; immutable but for the count. */
        struct Failure
        {
            /** @brief Constructs a failure that owns @p text, with one reference. */
            Failure(StatusCode failure_code, std::string&& text,
                    FailureBlockCounter* owner) noexcept
                : code(failure_code), owned(std::move(text)), message(owned), counter(owner)
            {

            }

            /** @brief Constructs a failure over @p literal, which must outlive it. */
            Failure(StatusCode failure_code, std::string_view literal) noexcept
                : code(failure_code), message(literal)
            {

            }

            /** @brief Counts the Status objects that share this block. */
            mutable std::atomic<std::uint32_t> count{1};

            /** @brief Names who has to act. */
            const StatusCode code;

            /** @brief Owns the text of a heap block; empty in the static block. */
            const std::string owned;

            /** @brief Views the text a caller reads: @c owned or a literal. */
            const std::string_view message;

            /** @brief Points at the counter that counted this block; null for the static one. */
            FailureBlockCounter* const counter = nullptr;
        };

        namespace
        {
            /** @brief Holds the message of the failure returned when no message could be built. */
            constexpr std::string_view K_EXHAUSTED_MESSAGE =
                "a failure's message could not be built: composing it threw, or the host heap "
                "refused the block that holds it";

            /** @brief Counts the heap failure blocks alive in the process. */
            class ProcessFailureBlockCounter final : public FailureBlockCounter
            {
                public:
                    /** @brief Counts one block made. */
                    void block_made() noexcept override
                    {
                        count_.fetch_add(1, std::memory_order_relaxed);
                    }

                    /** @brief Counts one block released. */
                    void block_released() noexcept override
                    {
                        count_.fetch_sub(1, std::memory_order_relaxed);
                    }

                    /** @brief Returns the blocks alive. */
                    [[nodiscard]] std::size_t live() const noexcept override
                    {
                        return count_.load(std::memory_order_relaxed);
                    }

                private:
                    /// @brief Holds the blocks alive.
                    std::atomic<std::size_t> count_{0};
            };

            /** @brief Returns the counter of the process, which is never destroyed. */
            ProcessFailureBlockCounter& process_counter() noexcept
            {
                static ProcessFailureBlockCounter counter;
                return counter;
            }

            /** @brief Returns the slot that holds the installed counter; it is never null. */
            std::atomic<FailureBlockCounter*>& installed_counter() noexcept
            {
                static std::atomic<FailureBlockCounter*> counter{&process_counter()};
                return counter;
            }

        } // namespace

        const Failure* allocate_failure(StatusCode code, std::string&& message) noexcept
        {
            FailureBlockCounter* const counter = installed_counter().load();
            const Failure* failure = new (std::nothrow) Failure(code, std::move(message), counter);
            if (failure != nullptr)
                counter->block_made();

            return failure;
        }

        const Failure* exhausted_failure() noexcept
        {
            // The static block holds its own reference, so its count never reaches zero.
            static const Failure exhausted(StatusCode::OutOfMemory, K_EXHAUSTED_MESSAGE);

            retain_failure(&exhausted);
            return &exhausted;
        }

        void retain_failure(const Failure* failure) noexcept
        {
            failure->count.fetch_add(1, std::memory_order_relaxed);
        }

        void release_failure(const Failure* failure) noexcept
        {
            if (failure->count.fetch_sub(1, std::memory_order_acq_rel) != 1)
                return;

            FailureBlockCounter* const counter = failure->counter;
            delete failure;
            if (counter != nullptr)
                counter->block_released();
        }

        std::size_t live_failure_count() noexcept
        {
            return process_counter().live();
        }

    } // namespace Detail

    FailureBlockCounter& set_failure_block_counter(FailureBlockCounter* counter) noexcept
    {
        FailureBlockCounter* const next =
            counter != nullptr ? counter : &Detail::process_counter();
        return *Detail::installed_counter().exchange(next);
    }

    StatusCode Status::code() const noexcept
    {
        if (failure_ == nullptr)
            return StatusCode::Ok;

        return failure_->code;
    }

    std::string_view Status::message() const noexcept
    {
        if (failure_ == nullptr)
            return std::string_view();

        return failure_->message;
    }

} // namespace SushiAI
