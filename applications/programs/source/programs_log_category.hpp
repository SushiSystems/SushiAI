/**************************************************************************/
/* programs_log_category.hpp                                              */
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
 * @file programs_log_category.hpp
 * @brief Declares K_PROGRAMS_LOG_CATEGORY, the log category of the programs.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiLog/log/log_category.hpp>
#include <SushiAI/log/sushiai_log_project.hpp>

namespace SushiAI
{
    namespace Apps
    {
        /** @brief Names the records of the programs, written as [sushiai.programs]. */
        inline constexpr SushiLog::LogCategory K_PROGRAMS_LOG_CATEGORY =
            SushiLog::make_log_category(K_SUSHIAI_LOG_PROJECT, "programs");
    } // namespace Apps
} // namespace SushiAI
