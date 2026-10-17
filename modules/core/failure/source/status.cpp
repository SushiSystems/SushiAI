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
 * @file status.cpp
 * @brief Implements the shared failure block behind Status and the accessors that read it.
 * @author Mustafa Garip
 */

#include <SushiAI/failure/status.hpp>

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
            Failure(StatusCode failure_code, std::string&& text) noexcept
                : code(failure_code), owned(std::move(text)), message(owned)
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
        };

        namespace
        {
            /** @brief Holds the message of the failure returned when no message could be built. */
            constexpr std::string_view K_EXHAUSTED_MESSAGE =
                "a failure's message could not be built: composing it threw, or the host heap "
                "refused the block that holds it";

            /** @brief Returns the count of heap failure blocks alive in the process. */
            std::atomic<std::size_t>& live_blocks() noexcept
            {
                static std::atomic<std::size_t> count{0};
                return count;
            }

        } // namespace

        const Failure* allocate_failure(StatusCode code, std::string&& message) noexcept
        {
            const Failure* failure = new (std::nothrow) Failure(code, std::move(message));
            if (failure != nullptr)
                live_blocks().fetch_add(1, std::memory_order_relaxed);

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

            live_blocks().fetch_sub(1, std::memory_order_relaxed);
            delete failure;
        }

        std::size_t live_failure_count() noexcept
        {
            return live_blocks().load(std::memory_order_relaxed);
        }

    } // namespace Detail

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
