/**************************************************************************/
/* result.hpp                                                             */
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
 * @file result.hpp
 * @brief Declares Result, a value or the Status that explains its absence.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    /**
     * @brief Holds a T, or the failure that kept one from being produced.
     *
     * Both constructors are implicit, against the single-argument rule, so that
     * @c return @c value; and @c return @c status; both work. There is no default
     * constructor and no Result of void; Status is that type.
     */
    template<typename T>
    class [[nodiscard]] Result final
    {
        static_assert(std::is_object<T>::value && !std::is_array<T>::value,
                      "Result<T> holds an object by value; use Result<T&> for a reference");
        static_assert(!std::is_same<std::remove_cv_t<T>, Status>::value,
                      "a Result of Status has two failures; return Status");

        public:
            /** @brief Holds @p value. */
            Result(T value) noexcept(std::is_nothrow_move_constructible<T>::value)
                : value_(std::in_place, std::move(value))
            {

            }

            /**
             * @brief Holds @p failure.
             * @param failure A failed Status; a success is fatal.
             */
            Result(Status failure) noexcept : status_(std::move(failure))
            {
                SA_FATAL_IF(status_.is_ok(),
                            "a Result was built from a success Status; it needs a value or a "
                            "failure");
            }

            /** @brief Reports whether a value is held. */
            [[nodiscard]] bool has_value() const noexcept
            {
                return value_.has_value();
            }

            /** @brief Returns the failure, or a success when a value is held. */
            [[nodiscard]] const Status& status() const noexcept
            {
                return status_;
            }

            /**
             * @brief Returns the held value.
             * @pre has_value(); a failure is fatal and reports its message.
             */
            [[nodiscard]] T& value() & noexcept
            {
                require_value();
                return *value_;
            }

            /**
             * @brief Returns the held value.
             * @pre has_value(); a failure is fatal and reports its message.
             */
            [[nodiscard]] const T& value() const& noexcept
            {
                require_value();
                return *value_;
            }

            /**
             * @brief Returns the held value for the caller to move from.
             * @pre has_value(); a failure is fatal and reports its message.
             */
            [[nodiscard]] T&& value() && noexcept
            {
                require_value();
                return std::move(*value_);
            }

        private:
            /** @brief Stops the process with the failure's message when no value is held. */
            void require_value() const noexcept
            {
                SA_FATAL_IF(!value_.has_value(),
                            "Result::value() was called on a failure: {}", status_.message());
            }

            /** @brief Holds the value; empty when a failure is held. */
            std::optional<T> value_;

            /** @brief Holds the failure; a success when a value is held. */
            Status status_;
    };

    /**
     * @brief Holds a reference to a T, or the failure that kept one from being found.
     *
     * The reference form of Result, with the same implicit constructors. It does not own
     * what it refers to.
     */
    template<typename T>
    class [[nodiscard]] Result<T&> final
    {
        static_assert(!std::is_same<std::remove_cv_t<T>, Status>::value,
                      "a Result of Status has two failures; return Status");

        public:
            /**
             * @brief Refers to @p value.
             * @param value Must outlive this Result and every reference taken from it.
             */
            Result(T& value) noexcept : target_(std::addressof(value))
            {

            }

            /** @brief Refuses a temporary, which would not outlive the Result. */
            Result(std::remove_const_t<T>&& value) = delete;

            /**
             * @brief Holds @p failure.
             * @param failure A failed Status; a success is fatal.
             */
            Result(Status failure) noexcept : status_(std::move(failure))
            {
                SA_FATAL_IF(status_.is_ok(),
                            "a Result was built from a success Status; it needs a value or a "
                            "failure");
            }

            /** @brief Reports whether a reference is held. */
            [[nodiscard]] bool has_value() const noexcept
            {
                return target_ != nullptr;
            }

            /** @brief Returns the failure, or a success when a reference is held. */
            [[nodiscard]] const Status& status() const noexcept
            {
                return status_;
            }

            /**
             * @brief Returns the held reference.
             * @pre has_value(); a failure is fatal and reports its message.
             */
            [[nodiscard]] T& value() const noexcept
            {
                SA_FATAL_IF(target_ == nullptr,
                            "Result::value() was called on a failure: {}", status_.message());
                return *target_;
            }

        private:
            /** @brief Points at the referent; null when a failure is held. */
            T* target_ = nullptr;

            /** @brief Holds the failure; a success when a reference is held. */
            Status status_;
    };

    namespace Detail
    {
        /** @brief Returns the status SA_TRY and SA_TRY_ASSIGN read from a Result. */
        template<typename T>
        [[nodiscard]] const Status& status_of(const Result<T>& result) noexcept
        {
            return result.status();
        }

    } // namespace Detail
} // namespace SushiAI

/** @brief Pastes two tokens without expanding them; the second step of SA_DETAIL_JOIN. */
#define SA_DETAIL_JOIN_TOKENS(left, right) left##right

/** @brief Pastes two tokens after expanding them, so __LINE__ becomes its number. */
#define SA_DETAIL_JOIN(left, right) SA_DETAIL_JOIN_TOKENS(left, right)

/** @brief Expands SA_TRY_ASSIGN over the local named @p outcome. */
#define SA_DETAIL_TRY_ASSIGN(outcome, declaration, expression)                         \
    auto&& outcome = (expression);                                                     \
    if (!::SushiAI::Detail::status_of(outcome).is_ok())                                \
    {                                                                                  \
        return ::SushiAI::Detail::status_of(outcome);                                  \
    }                                                                                  \
    declaration = static_cast<decltype(outcome)>(outcome).value()

/**
 * @brief Binds the value of @p expression to @p declaration, or returns its failure.
 * @param declaration A declaration with no top-level comma, such as @c const @c Shape @c out.
 * @param expression A Result or a Graph::Traced, as an lvalue or a prvalue and never a
 *                   reference into a temporary; evaluated once, and moved from when a prvalue.
 * @pre The macro is a statement of its own: it expands to more than one statement, so it
 *      cannot be the body of an unbraced @c if, and one line holds at most one.
 */
#define SA_TRY_ASSIGN(declaration, expression)                                         \
    SA_DETAIL_TRY_ASSIGN(SA_DETAIL_JOIN(sa_try_outcome_, __LINE__), declaration, expression)
