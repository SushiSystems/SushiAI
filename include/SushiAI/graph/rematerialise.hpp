/**************************************************************************/
/* rematerialise.hpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                https://github.com/SushiSystems/SushiAI                 */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License");        */
/* you may not use this file except in compliance with the License.       */
/* You may obtain a copy of the License at                                */
/*                                                                        */
/*     http://www.apache.org/licenses/LICENSE-2.0                         */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file rematerialise.hpp
 * @brief Declares the rematerialise_im2col pass for recomputing column matrices.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Graph
    {
        /**
         * @brief Recomputes every stored @c IM2COL result at its later consumers.
         *
         * Inserts gathers before later readers to shorten column matrix live ranges.
         *
         * @param graph The graph to rewrite in place.
         */
        void rematerialise_im2col(Graph& graph);

    } // namespace Graph
} // namespace SushiAI
