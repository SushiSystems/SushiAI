/**************************************************************************/
/* replay_prelude.hpp                                                     */
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
 * @file replay_prelude.hpp
 * @brief Declares ReplayPrelude, the host work a lowered node needs before each replay.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <functional>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Names host work a lowered node needs before each replay of its step.
         *
         * It is called on the thread that runs the step, with the number of replays the step
         * has finished, which is 0 before the first. It records no device work, reads no
         * device result, allocates nothing and does not throw.
         */
        using ReplayPrelude = std::function<void(std::uint64_t run)>;

    } // namespace Graph
} // namespace SushiAI
