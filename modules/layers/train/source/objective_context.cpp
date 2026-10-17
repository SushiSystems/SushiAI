/**************************************************************************/
/* objective_context.cpp                                                  */
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
 * @file objective_context.cpp
 * @brief Implements the listing of the extents a traced model published.
 * @author Mustafa Garip
 */

#include <SushiAI/train/objective_context.hpp>

#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        Result<std::vector<OutputShape>> list_output_shapes(const Graph::Graph& graph,
                                                            const NN::ModelOutputs& outputs)
        {
            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<OutputShape> listed;
                listed.reserve(outputs.all().size());
                for (const NN::ModelOutputs::Entry& output : outputs.all())
                    listed.push_back(OutputShape{output.first, graph.value(output.second).shape});

                return listed;
            });
        }

        const OutputShape* find_output_shape(const std::vector<OutputShape>& outputs,
                                             std::string_view name) noexcept
        {
            for (const OutputShape& output : outputs)
            {
                if (output.name == name)
                    return &output;
            }

            return nullptr;
        }

    } // namespace Train
} // namespace SushiAI
