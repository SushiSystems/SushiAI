/**************************************************************************/
/* domains_language_none.cpp                                              */
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
 * @file domains_language_none.cpp
 * @brief Adds no row of language, for a build that does not link it.
 * @author Mustafa Garip
 */

#include "domains.hpp"

#include <SushiAI/core/status.hpp>

namespace SushiAI
{
    namespace Apps
    {
        Status register_language_domain(Registries&)
        {
            return Status();
        }

    } // namespace Apps
} // namespace SushiAI
