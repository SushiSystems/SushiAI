/**************************************************************************/
/* sushiai_log_project.hpp                                                */
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
 * @file sushiai_log_project.hpp
 * @brief Declares K_SUSHIAI_LOG_PROJECT, the log project of SushiAI.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiLog/log/log_project.hpp>

namespace SushiAI
{
    /** @brief Names SushiAI's records, whose levels SA_LOG_LEVEL sets. */
    inline constexpr SushiLog::LogProject K_SUSHIAI_LOG_PROJECT{"sushiai", "SA_LOG_LEVEL"};
} // namespace SushiAI
