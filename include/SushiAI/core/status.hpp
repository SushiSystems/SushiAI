/**************************************************************************/
/* status.hpp                                                             */
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
 * @file status.hpp
 * @brief Declares Status, its codes and the macros that return or forward a failure.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <SushiBLAS/core/format.hpp>
#include <SushiAI/core/check.hpp>

namespace SushiAI
{
    /** @brief Names who has to act on a failure. */
    enum class StatusCode : std::uint8_t
    {
        Ok,                 /**< Nothing failed. */
        InvalidDescription, /**< A document or a traced model is wrong; fix the description. */
        FileNotFound,       /**< A named file is absent. */
        InvalidFile,        /**< A file is unreadable, truncated or of the wrong version. */
        Incompatible,       /**< Two valid things disagree, such as a checkpoint and a model. */
        Unsupported,        /**< This build, device or execution mode lacks the capability. */
        OutOfMemory,        /**< A device, arena or failure-message allocation was refused. */
        DeviceFailed,       /**< A kernel faulted, the device was lost or a task threw. */
    };

    /**
     * @brief Returns a lower-case name for @p code, such as "invalid_description".
     * @return A null-terminated literal owned by the library; never null.
     */
    [[nodiscard]] constexpr const char* status_code_name(StatusCode code) noexcept
    {
        switch (code)
        {
            case StatusCode::Ok:                 return "ok";
            case StatusCode::InvalidDescription: return "invalid_description";
            case StatusCode::FileNotFound:       return "file_not_found";
            case StatusCode::InvalidFile:        return "invalid_file";
            case StatusCode::Incompatible:       return "incompatible";
            case StatusCode::Unsupported:        return "unsupported";
            case StatusCode::OutOfMemory:        return "out_of_memory";
            case StatusCode::DeviceFailed:       return "device_failed";
        }
        return "unknown";
    }

    namespace Detail
    {
        /** @brief Holds one failure's count, code and text; defined in status.cpp. */
        struct Failure;

        /**
         * @brief Allocates a failure block that owns @p message and carries one reference.
         * @return The block, or a null pointer when the host heap refused it.
         */
        [[nodiscard]] const Failure* allocate_failure(StatusCode code,
                                                      std::string&& message) noexcept;

        /** @brief Returns the static OutOfMemory failure with one more reference on it. */
        [[nodiscard]] const Failure* exhausted_failure() noexcept;

        /** @brief Adds one reference to @p failure. */
        void retain_failure(const Failure* failure) noexcept;

        /** @brief Drops one reference from @p failure and frees the block at zero. */
        void release_failure(const Failure* failure) noexcept;

        /** @brief Returns how many heap failure blocks are alive in the process. */
        [[nodiscard]] std::size_t live_failure_count() noexcept;

    } // namespace Detail

    /**
     * @brief Holds success, or a failure code with a message a user can act on.
     *
     * Success is one null pointer. A failure is one shared block that never changes, so
     * copies of it may live on several threads.
     */
    class [[nodiscard]] Status final
    {
        public:
            /** @brief Constructs success, which allocates nothing. */
            Status() noexcept = default;

            /** @brief Shares the failure of @p other, or copies its success. */
            Status(const Status& other) noexcept : failure_(other.failure_)
            {
                if (failure_ != nullptr)
                    Detail::retain_failure(failure_);
            }

            /** @brief Shares the failure of @p other, which keeps it. */
            Status(Status&& other) noexcept : Status(static_cast<const Status&>(other))
            {

            }

            /** @brief Replaces this status with a share of @p other. */
            Status& operator=(const Status& other) noexcept
            {
                Status copy(other);
                std::swap(failure_, copy.failure_);
                return *this;
            }

            /** @brief Replaces this status with a share of @p other, which keeps its failure. */
            Status& operator=(Status&& other) noexcept
            {
                return *this = static_cast<const Status&>(other);
            }

            /** @brief Drops this status's share of its failure. */
            ~Status()
            {
                if (failure_ != nullptr)
                    Detail::release_failure(failure_);
            }

            /**
             * @brief Builds a failure whose message @p compose returns.
             * @param code Any code but Ok; Ok is fatal.
             * @param compose Called once, inside this function; returns std::string.
             * @return The failure, or the static OutOfMemory failure when @p compose throws
             *         or the block cannot be allocated.
             */
            template<typename Compose>
            [[nodiscard]] static Status from_message(StatusCode code, Compose&& compose) noexcept;

            /**
             * @brief Returns a copy whose message is "<prefix>: <message>".
             * @param compose Returns the prefix as std::string; not called on success.
             * @return The annotated failure, or this status unchanged when it is a success,
             *         @p compose throws or the block cannot be allocated.
             */
            template<typename Compose>
            [[nodiscard]] Status annotated(Compose&& compose) const noexcept;

            /** @brief Reports whether this status is a success. */
            [[nodiscard]] bool is_ok() const noexcept
            {
                return failure_ == nullptr;
            }

            /** @brief Returns the failure's code, or Ok. */
            [[nodiscard]] StatusCode code() const noexcept;

            /** @brief Returns the failure's message; empty on success. */
            [[nodiscard]] std::string_view message() const noexcept;

        private:
            /** @brief Adopts one reference to @p failure. */
            explicit Status(const Detail::Failure* failure) noexcept : failure_(failure)
            {

            }

            /** @brief Holds the shared failure block; null on success. */
            const Detail::Failure* failure_ = nullptr;
    };

    template<typename Compose>
    Status Status::from_message(StatusCode code, Compose&& compose) noexcept
    {
        SA_FATAL_IF(code == StatusCode::Ok,
                    "Status::from_message needs a failure code; got {}", status_code_name(code));

        try
        {
            std::string message = std::forward<Compose>(compose)();
            const Detail::Failure* failure = Detail::allocate_failure(code, std::move(message));
            if (failure != nullptr)
                return Status(failure);

            return Status(Detail::exhausted_failure());
        }
        catch (...)
        {
            return Status(Detail::exhausted_failure());
        }
    }

    template<typename Compose>
    Status Status::annotated(Compose&& compose) const noexcept
    {
        if (is_ok())
            return Status();

        try
        {
            std::string text = std::forward<Compose>(compose)();
            text.append(": ");
            text.append(message());

            const Detail::Failure* failure = Detail::allocate_failure(code(), std::move(text));
            if (failure != nullptr)
                return Status(failure);

            return *this;
        }
        catch (...)
        {
            return *this;
        }
    }

    namespace Detail
    {
        /** @brief Returns the status SA_TRY and SA_TRY_ASSIGN read from a Status. */
        [[nodiscard]] inline const Status& status_of(const Status& status) noexcept
        {
            return status;
        }

    } // namespace Detail
} // namespace SushiAI

/**
 * @brief Returns a failed Status with a formatted message when @p condition holds.
 * @param condition Expression to evaluate; the same polarity as SA_THROW_IF.
 * @param code The StatusCode of the failure.
 * @param ... Format string and arguments for SushiBLAS::Core::format, evaluated only on failure.
 */
#define SA_FAIL_IF(condition, code, ...)                                               \
    do                                                                                 \
    {                                                                                  \
        if (condition)                                                                 \
        {                                                                              \
            return ::SushiAI::Status::from_message((code), [&]()                       \
            {                                                                          \
                return ::SushiBLAS::Core::format(__VA_ARGS__);                         \
            });                                                                        \
        }                                                                              \
    } while (0)

/**
 * @brief Yields @p status annotated with a formatted prefix.
 * @param status A Status expression.
 * @param ... Format string and arguments for SushiBLAS::Core::format; not evaluated on success.
 */
#define SA_ANNOTATE(status, ...)                                                       \
    (status).annotated([&]()                                                           \
    {                                                                                  \
        return ::SushiBLAS::Core::format(__VA_ARGS__);                                 \
    })

/**
 * @brief Returns the failure of @p expression from the enclosing function, if it holds one.
 * @param expression A Status, a Result or a Graph::Traced, or a reference to one, which may
 *                   point into a temporary; evaluated once.
 */
#define SA_TRY(expression)                                                             \
    do                                                                                 \
    {                                                                                  \
        const ::SushiAI::Status sa_try_status_(                                        \
            ::SushiAI::Detail::status_of((expression)));                               \
        if (!sa_try_status_.is_ok())                                                   \
        {                                                                              \
            return sa_try_status_;                                                     \
        }                                                                              \
    } while (0)
