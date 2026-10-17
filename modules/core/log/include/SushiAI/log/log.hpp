/**************************************************************************/
/* log.hpp                                                                */
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
 * @file log.hpp
 * @brief Declares SA_LOG_TRACE through SA_LOG_FATAL, SushiAI's front on SushiLog.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiLog/log/log_front.hpp>
#include <SushiLog/log/log_level.hpp>

#ifndef SA_ENABLE_LOGGING
    #define SA_ENABLE_LOGGING 1
#endif

namespace SushiAI
{
    /** @brief Holds the least severe level this translation unit compiles SushiAI log calls for. */
    [[maybe_unused]] constexpr SushiLog::LogLevel K_SUSHIAI_LOG_CEILING =
        SushiLog::log_level_ceiling(SA_ENABLE_LOGGING != 0);
} // namespace SushiAI

/** @brief Logs at @p level in @p category through this translation unit's ceiling. */
#define SA_LOG_AT(level, category, ...)                                                          \
    SL_LOG_FRONT_AT(SushiAI::K_SUSHIAI_LOG_CEILING, level, category, __VA_ARGS__)

/** @brief Logs step-by-step flow in a category; each {} takes the next argument. */
#define SA_LOG_TRACE(category, ...) SA_LOG_AT(SushiLog::LogLevel::Trace, category, __VA_ARGS__)

/** @brief Logs internal state in a category; each {} takes the next argument. */
#define SA_LOG_DEBUG(category, ...) SA_LOG_AT(SushiLog::LogLevel::Debug, category, __VA_ARGS__)

/** @brief Logs a lifecycle event in a category; each {} takes the next argument. */
#define SA_LOG_INFO(category, ...) SA_LOG_AT(SushiLog::LogLevel::Info, category, __VA_ARGS__)

/** @brief Logs a recovery or a fallback in a category; each {} takes the next argument. */
#define SA_LOG_WARN(category, ...) SA_LOG_AT(SushiLog::LogLevel::Warn, category, __VA_ARGS__)

/** @brief Logs a failed operation in a category; each {} takes the next argument. */
#define SA_LOG_ERROR(category, ...) SA_LOG_AT(SushiLog::LogLevel::Error, category, __VA_ARGS__)

/** @brief Logs why the process stops, delivered and flushed before the call returns. */
#define SA_LOG_FATAL(category, ...) SA_LOG_AT(SushiLog::LogLevel::Fatal, category, __VA_ARGS__)
