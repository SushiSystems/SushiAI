/**************************************************************************/
/* run_progress.cpp                                                       */
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
 * @file run_progress.cpp
 * @brief Implements the guarded call of a detection run's progress callback.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/run_progress.hpp>

#include <SushiAI/seam/seam.hpp>

namespace SushiAI::Vision::Training
{
    Status notify_guarded(const std::function<void()>& notify) noexcept
    {
        if (!notify)
            return Status();

        return ::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Execute, [&]()
        {
            notify();
            return Status();
        });
    }

} // namespace SushiAI::Vision::Training
