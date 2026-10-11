/**************************************************************************/
/* segment.hpp                                                            */
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
 * @file segment.hpp
 * @brief Declares Segment, one run of consecutively traced nodes under one scope.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstdint>
#include <string>

#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Eager
    {
        /**
         * @brief Names a maximal run of consecutively traced nodes with one innermost scope.
         * @see modules/core/eager/README.md
         */
        struct Segment
        {
            std::string name;             /**< Scope path, "outer/inner"; empty outside any scope. */
            std::uint32_t occurrence = 0; /**< Index among the trace's segments of this name. */
            Graph::NodeID begin = 0;      /**< First node of the run. */
            Graph::NodeID end = 0;        /**< One past the last node of the run. */
        };

    } // namespace Eager
} // namespace SushiAI
