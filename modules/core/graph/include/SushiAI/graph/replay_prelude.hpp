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
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
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
