/**************************************************************************/
/* built_rows.hpp                                                         */
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
 * @file built_rows.hpp
 * @brief Declares BuiltRows, the view a row's factory has of the nodes built above it.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <SushiAI/nn/node.hpp>

namespace SushiAI
{
    namespace NN
    {
        class ModelGraph;

        /**
         * @brief Views the nodes of the rows above the one being built, by row index.
         *
         * It owns nothing. A node it returns lives as long as the graph it was built into and
         * stays at its address when that graph moves.
         */
        class BuiltRows final
        {
            public:
                /** @brief Views no row, which is what row 0 and a node built alone see. */
                BuiltRows() noexcept = default;

                /**
                 * @brief Views every node of @p graph.
                 * @param graph A graph whose node i is row i; it must outlive this view.
                 */
                explicit BuiltRows(const ModelGraph& graph) noexcept;

                /** @brief Returns how many rows are above, which is the index being built. */
                [[nodiscard]] std::size_t size() const noexcept;

                /**
                 * @brief Returns the node of row @p index.
                 * @pre index < size(); anything else is fatal.
                 */
                [[nodiscard]] Node& node(std::size_t index) const;

            private:
                /** @brief Points at the graph the rows were added to; null views no row. */
                const ModelGraph* graph_ = nullptr;
        };

    } // namespace NN
} // namespace SushiAI
