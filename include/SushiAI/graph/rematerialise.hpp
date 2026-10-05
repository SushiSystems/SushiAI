/****************************************************************/
/* rematerialise.hpp                                            */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

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
