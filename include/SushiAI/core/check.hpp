/**************************************************************************/
/* check.hpp                                                              */
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
 * @file check.hpp
 * @brief Declares the fatal checks and the handler a fatal report is delivered to.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <SushiBLAS/core/format.hpp>

namespace SushiAI
{
    /** @brief Receives one fatal report; the process stops after it returns. */
    using FatalHandler = void (*)(const char* file, int line, std::string_view message) noexcept;

    /**
     * @brief Installs @p handler as the receiver of fatal reports.
     * @param handler The receiver; a null pointer installs the one that writes to stderr.
     * @return The handler that was installed before the call; never null.
     */
    FatalHandler set_fatal_handler(FatalHandler handler) noexcept;

    namespace Detail
    {
        /** @brief Holds the text a fatal check reports when composing its own message threw. */
        inline constexpr std::string_view K_UNCOMPOSED_FATAL_MESSAGE =
            "a fatal check failed and its message could not be composed";

        /** @brief Delivers one report to the installed handler and returns. */
        void report_fatal(const char* file, int line, std::string_view message) noexcept;

        /** @brief Delivers the report, flushes every output stream and aborts the process. */
        [[noreturn]] void fail_fast(const char* file, int line, std::string_view message) noexcept;

        /**
         * @brief Stops the process through fail_fast with the text @p compose returns.
         * @param compose Called once; returns std::string. A throw from it is reported as
         *                K_UNCOMPOSED_FATAL_MESSAGE.
         */
        template<typename Compose>
        [[noreturn]] void fail_fast_composed(const char* file, int line,
                                             Compose&& compose) noexcept
        {
            try
            {
                const std::string message = std::forward<Compose>(compose)();
                fail_fast(file, line, message);
            }
            catch (...)
            {
                fail_fast(file, line, K_UNCOMPOSED_FATAL_MESSAGE);
            }
        }

    } // namespace Detail
} // namespace SushiAI

/**
 * @brief Stops the process with a formatted message when @p condition holds, in every build.
 * @param condition Expression to evaluate; the same polarity as SA_THROW_IF.
 * @param ... Format string and arguments for SushiBLAS::Core::format, evaluated only on failure.
 */
#define SA_FATAL_IF(condition, ...)                                                    \
    do                                                                                 \
    {                                                                                  \
        if (condition)                                                                 \
        {                                                                              \
            ::SushiAI::Detail::fail_fast_composed(__FILE__, __LINE__, [&]()            \
            {                                                                          \
                return ::SushiBLAS::Core::format(__VA_ARGS__);                         \
            });                                                                        \
        }                                                                              \
    } while (0)

/**
 * @brief Stops the process with a formatted message, in every build.
 * @param ... Format string and arguments for SushiBLAS::Core::format.
 */
#define SA_FATAL(...)                                                                  \
    ::SushiAI::Detail::fail_fast_composed(__FILE__, __LINE__, [&]()                    \
    {                                                                                  \
        return ::SushiBLAS::Core::format(__VA_ARGS__);                                 \
    })

#ifdef NDEBUG

/**
 * @brief Stops the process when @p condition is false, in debug builds only.
 * @param condition Expression with no side effect; not evaluated when NDEBUG is defined.
 */
#define SA_ASSERT(condition)                                                           \
    do                                                                                 \
    {                                                                                  \
        (void)sizeof(!(condition));                                                    \
    } while (0)

#else

/**
 * @brief Stops the process when @p condition is false, in debug builds only.
 * @param condition Expression with no side effect; not evaluated when NDEBUG is defined.
 */
#define SA_ASSERT(condition)                                                           \
    do                                                                                 \
    {                                                                                  \
        if (!(condition))                                                              \
        {                                                                              \
            ::SushiAI::Detail::fail_fast(__FILE__, __LINE__,                           \
                                         "assertion failed: " #condition);             \
        }                                                                              \
    } while (0)

#endif
