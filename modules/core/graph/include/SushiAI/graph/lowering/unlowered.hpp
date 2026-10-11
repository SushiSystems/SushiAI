/**************************************************************************/
/* unlowered.hpp                                                          */
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
 * @file unlowered.hpp
 * @brief Declares the list of operations that have no lowering row.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <SushiAI/core/ops.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /** @brief Lists the operations recorded beside the graph, which no node carries. */
        inline constexpr std::array<OpID, 4> K_UNLOWERED = {{
            Ops::K_ADAMW_UPDATE.id,
            Ops::K_SGD_UPDATE.id,
            Ops::K_CLIP_SCALE.id,
            Ops::K_BATCHNORM_UPDATE_STATS.id,
        }};

    } // namespace Graph
} // namespace SushiAI
