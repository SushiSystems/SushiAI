/****************************************************************/
/* test_throws.hpp                                              */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

#pragma once

/**
 * @file test_throws.hpp
 * @brief Asserting that a failure is loud @e and says what is wrong.
 *
 * @c EXPECT_THROW proves only that something went wrong, which is the weaker
 * half of the contract. Every @c SA_THROW_IF in this library names the actual
 * problem - the two shapes that disagree, the operation with no rule, the
 * parameter nothing can move - and that naming is the part a reader depends on
 * at three in the morning. It is also the part a refactor silently destroys:
 * collapsing five specific diagnostics into one generic "invalid argument"
 * keeps every @c EXPECT_THROW passing.
 *
 * So the assertions below take the substrings the message must contain. A
 * diagnostic that stops naming the offending value fails the test that covers
 * it, rather than being noticed a year later by whoever hits it.
 *
 * Host lane: no SushiBLAS, no SYCL, so the device-free unit tier can use it
 * too.
 */

#include <exception>
#include <initializer_list>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @struct ThrowOutcome
         * @brief What running a statement actually did.
         */
        struct ThrowOutcome
        {
            /** @brief It threw @c SushiAI::Error. */
            bool threw_error = false;

            /** @brief It threw something that is not @c SushiAI::Error. */
            bool threw_other = false;

            /** @brief The Error's message, when @c threw_error. */
            std::string message;

            /** @brief The foreign exception's text, when @c threw_other. */
            std::string other_what;
        };

        /**
         * @brief Runs @p action and reports how it failed, if it did.
         *
         * A foreign exception is recorded rather than propagated so the
         * assertion can say "threw std::bad_alloc instead of SushiAI::Error",
         * which is a far more useful failure than an aborted test.
         *
         * @param action The statement to run.
         * @return What happened.
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
 * @brief Asserts @p statement throws SushiAI::Error whose message names each
 *        of the remaining arguments.
 *
 * Parenthesise @p statement when it contains a top-level comma - the
 * preprocessor honours parentheses but not braces, so
 * @c SA_EXPECT_ERROR((Shape{1, 2}), "...") is required where
 * @c SA_EXPECT_ERROR(Shape{1, 2}, "...") would split into three arguments.
 */
#define SA_EXPECT_ERROR(statement, ...)                                        \
    EXPECT_TRUE(::SushiAI::Test::error_names(                                  \
        ::SushiAI::Test::capture_error([&]() { statement; }), {__VA_ARGS__}))   \
        << "  statement: " #statement
