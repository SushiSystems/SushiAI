/**************************************************************************/
/* run_progress.hpp                                                       */
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

/**
 * @file run_progress.hpp
 * @brief Declares the delivery of one update's progress to the callback of a detection run.
 * @author Mustafa Garip
 */

#pragma once

#include <functional>

#include <SushiAI/failure/status.hpp>

namespace SushiAI::Vision::Training
{
    /**
     * @brief Calls @p notify and turns a throw of it into a failure.
     * @param notify The call to make; an empty function succeeds without a call.
     * @return Success, or the failure the throw of @p notify became.
     */
    [[nodiscard]] Status notify_guarded(const std::function<void()>& notify) noexcept;

    /**
     * @brief Hands @p progress to @p callback, when it holds one, and guards its throw.
     * @tparam Progress The progress type of one detection engine.
     * @param callback Receives the progress; may be empty.
     * @param progress What one update produced.
     * @return Success, or the failure the throw of @p callback became.
     */
    template <typename Progress>
    [[nodiscard]] Status report_run_progress(
        const std::function<void(const Progress&)>& callback, const Progress& progress) noexcept
    {
        if (!callback)
            return Status();

        return notify_guarded([&callback, &progress]() { callback(progress); });
    }

} // namespace SushiAI::Vision::Training
