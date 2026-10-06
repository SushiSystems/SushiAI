/**************************************************************************/
/* rematerialise.hpp                                                      */
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
