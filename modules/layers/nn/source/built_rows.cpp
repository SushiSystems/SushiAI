/**************************************************************************/
/* built_rows.cpp                                                         */
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
 * @file built_rows.cpp
 * @brief Implements BuiltRows over the nodes of a model graph.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/built_rows.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/nn/model_graph.hpp>

namespace SushiAI
{
    namespace NN
    {
        BuiltRows::BuiltRows(const ModelGraph& graph) noexcept : graph_(&graph)
        {

        }

        std::size_t BuiltRows::size() const noexcept
        {
            return graph_ == nullptr ? 0 : graph_->size();
        }

        Node& BuiltRows::node(std::size_t index) const noexcept
        {
            SA_FATAL_IF(index >= size(), "BuiltRows::node was asked for row {}; {} rows are above",
                        index, size());

            return graph_->node(index);
        }

    } // namespace NN
} // namespace SushiAI
