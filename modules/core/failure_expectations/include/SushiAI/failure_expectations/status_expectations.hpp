/**************************************************************************/
/* status_expectations.hpp                                                */
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
 * @file status_expectations.hpp
 * @brief Declares assertions that check the code of a Status and what its message names.
 * @author Mustafa Garip
 */

#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Checks that @p status is a success, or fails quoting its code and message. */
        inline ::testing::AssertionResult status_is_ok(const Status& status) noexcept
        {
            if (status.is_ok())
                return ::testing::AssertionSuccess();

            return ::testing::AssertionFailure()
                << "expected success, got " << status_code_name(status.code()) << ".\n"
                << "  message: " << status.message();
        }

        /**
         * @brief Checks that @p status is a failure of @p expected naming every one of @p needles.
         * @param needles Substrings the message must contain, in any order.
         * @return Success, or a failure quoting the code and the whole message.
         */
        inline ::testing::AssertionResult status_names(
            const Status& status, StatusCode expected,
            std::initializer_list<std::string_view> needles) noexcept
        {
            if (status.is_ok())
                return ::testing::AssertionFailure()
                    << "expected " << status_code_name(expected) << ", got success";

            if (status.code() != expected)
                return ::testing::AssertionFailure()
                    << "expected " << status_code_name(expected) << ", got "
                    << status_code_name(status.code()) << ".\n"
                    << "  message: " << status.message();

            for (std::string_view needle : needles)
                if (status.message().find(needle) == std::string_view::npos)
                    return ::testing::AssertionFailure()
                        << "the diagnostic does not name \"" << needle << "\".\n"
                        << "  message: " << status.message();

            return ::testing::AssertionSuccess();
        }

        /** @brief Forwards a pack of needles to status_names, so a macro may pass none. */
        template<typename... Needles>
        ::testing::AssertionResult failure_names(const Status& status, StatusCode expected,
                                                 const Needles&... needles) noexcept
        {
            return status_names(status, expected, {std::string_view(needles)...});
        }

        /**
         * @brief Joins @p needles with ".*" and escapes regular-expression characters.
         * @param needles Substrings in the order the fatal message prints them.
         * @return A pattern for a death test, valid in POSIX and in GoogleTest's own syntax.
         */
        inline std::string death_pattern(std::initializer_list<std::string_view> needles) noexcept
        {
            constexpr std::string_view K_SPECIAL = "\\.*+?^$()[]{}|";

            std::string pattern;
            for (std::string_view needle : needles)
            {
                if (!pattern.empty())
                    pattern += ".*";

                for (char character : needle)
                {
                    if (K_SPECIAL.find(character) != std::string_view::npos)
                        pattern += '\\';

                    pattern += character;
                }
            }

            return pattern;
        }

        /**
         * @brief Returns the value of @p outcome, or stops the process with the failure it holds.
         *
         * For a test helper that returns a value and so cannot use SA_ASSERT_OK_AND_ASSIGN: a
         * fixture the test wrote is a precondition, and a refusal of it is a defect of the test.
         * A test of a refusal expects the failure itself.
         */
        template<typename T>
        [[nodiscard]] T require_value(Result<T> outcome) noexcept
        {
            SA_FATAL_IF(!outcome.has_value(), "the test's fixture is refused: {}",
                        outcome.status().message());

            // The value is held, so value() succeeds.
            return std::move(outcome).value();
        }

        /** @brief Returns the reference @p outcome holds, or stops the process with its failure. */
        template<typename T>
        [[nodiscard]] T& require_value(Result<T&> outcome) noexcept
        {
            SA_FATAL_IF(!outcome.has_value(), "the test's fixture is refused: {}",
                        outcome.status().message());

            // The reference is held, so value() succeeds.
            return outcome.value();
        }

    } // namespace Test
} // namespace SushiAI

/**
 * @brief Expects @p expression to be a failure with the given code naming each needle.
 * @param expression A Status, a Result or a Graph::Traced; parenthesised when it contains a
 *                   top-level comma.
 * @param ... The StatusCode, then any number of substrings the message must contain.
 */
#define SA_EXPECT_FAILURE(expression, ...)                                             \
    EXPECT_TRUE(::SushiAI::Test::failure_names(                                        \
        ::SushiAI::Detail::status_of(expression), __VA_ARGS__))                        \
        << "  expression: " #expression

/**
 * @brief Expects @p expression to be a success, and prints its message when it is not.
 * @param expression A Status, a Result or a Graph::Traced.
 */
#define SA_EXPECT_OK(expression)                                                       \
    EXPECT_TRUE(::SushiAI::Test::status_is_ok(::SushiAI::Detail::status_of(expression)))  \
        << "  expression: " #expression

/** @brief Expands SA_ASSERT_OK_AND_ASSIGN over the local named @p outcome. */
#define SA_DETAIL_ASSERT_OK_AND_ASSIGN(outcome, declaration, expression)               \
    auto&& outcome = (expression);                                                     \
    ASSERT_TRUE(::SushiAI::Test::status_is_ok(::SushiAI::Detail::status_of(outcome)))  \
        << "  expression: " #expression;                                               \
    declaration = static_cast<decltype(outcome)>(outcome).value()

/**
 * @brief Binds the value of @p expression to @p declaration, or fails this test with the message.
 * @param declaration A declaration with no top-level comma.
 * @param expression A Result or a Graph::Traced; evaluated once.
 * @pre The enclosing function returns void, and one line holds at most one use.
 */
#define SA_ASSERT_OK_AND_ASSIGN(declaration, expression)                               \
    SA_DETAIL_ASSERT_OK_AND_ASSIGN(SA_DETAIL_JOIN(sa_assert_outcome_, __LINE__),       \
                                   declaration, expression)

/**
 * @brief Expects @p statement to stop the process with a fatal report naming each needle.
 * @param statement The statement to run in the death-test child.
 * @param ... One or more substrings, in the order the report prints them.
 */
#define SA_EXPECT_FATAL(statement, ...)                                                \
    EXPECT_DEATH(statement, ::SushiAI::Test::death_pattern({__VA_ARGS__}))
