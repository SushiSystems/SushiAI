/**************************************************************************/
/* throw_expectations.hpp                                                 */
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
 * @file throw_expectations.hpp
 * @brief Declares assertions that check what the message of a thrown Error names.
 * @author Mustafa Garip
 */

#pragma once

#include <exception>
#include <initializer_list>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/failure/error.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds what running a statement did. */
        struct ThrowOutcome
        {
            /** @brief Holds whether it threw @c SushiAI::Error. */
            bool threw_error = false;

            /** @brief Holds whether it threw something that is not @c SushiAI::Error. */
            bool threw_other = false;

            /** @brief Holds the Error's message, when @c threw_error. */
            std::string message;

            /** @brief Holds the foreign exception's text, when @c threw_other. */
            std::string other_what;
        };

        /**
         * @brief Runs @p action and reports how it failed, if it did.
         */
        template<typename Fn>
        ThrowOutcome capture_error(Fn&& action)
        {
            try
            {
                action();
            }
            catch (const Error& error)
            {
                return ThrowOutcome{true, false, error.what(), {}};
            }
            catch (const std::exception& error)
            {
                return ThrowOutcome{false, true, {}, error.what()};
            }
            catch (...)
            {
                return ThrowOutcome{false, true, {}, "an exception not derived from std::exception"};
            }

            return ThrowOutcome{};
        }

        /**
         * @brief Checks that @p outcome is an Error naming every one of @p required.
         *
         * @param outcome What @c capture_error saw.
         * @param required Substrings the message must contain, in any order.
         * @return Success, or a failure quoting the whole message.
         */
        inline ::testing::AssertionResult error_names(const ThrowOutcome& outcome,
                                                      std::initializer_list<std::string_view> required)
        {
            if (outcome.threw_other)
                return ::testing::AssertionFailure()
                    << "threw something other than SushiAI::Error: " << outcome.other_what;

            if (!outcome.threw_error)
                return ::testing::AssertionFailure() << "did not throw SushiAI::Error at all";

            for (std::string_view needle : required)
                if (outcome.message.find(needle) == std::string::npos)
                    return ::testing::AssertionFailure()
                        << "the diagnostic no longer names \"" << needle << "\".\n"
                        << "  message: " << outcome.message;

            return ::testing::AssertionSuccess();
        }

    } // namespace Test
} // namespace SushiAI

/**
 * @brief Asserts @p statement throws SushiAI::Error whose message names each remaining argument.
 * @pre @p statement is parenthesised when it contains a top-level comma.
 */
#define SA_EXPECT_ERROR(statement, ...)                                        \
    EXPECT_TRUE(::SushiAI::Test::error_names(                                  \
        ::SushiAI::Test::capture_error([&]() { statement; }), {__VA_ARGS__}))   \
        << "  statement: " #statement
