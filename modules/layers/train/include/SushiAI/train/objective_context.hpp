/**************************************************************************/
/* objective_context.hpp                                                  */
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
 * @file objective_context.hpp
 * @brief Declares what an objective factory may read besides its own section.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <SushiAI/config/value.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>

namespace SushiAI
{
    namespace Train
    {
        class ObjectiveRegistry;

        /** @brief Pairs the name of a published model output with its extents, batch first. */
        struct OutputShape
        {
            std::string name; /**< Name the model publishes the output under. */
            Shape shape;      /**< Extents of the output, the batch axis first. */
        };

        /** @brief Holds what an objective factory may read besides its own section. */
        struct ObjectiveContext
        {
            std::int64_t batch = 0;           /**< Samples per step. */
            std::int64_t classes = 0;         /**< Classes of the data; 0 when it states none. */
            Shape input_sample;               /**< Shape of one sample entering the model. */
            Config::Value task;               /**< Task section of the model; null for none. */
            std::vector<OutputShape> outputs; /**< Published outputs, in published order. */
            /// @brief The registry a composite builds its terms from; make_objective() sets it.
            const ObjectiveRegistry* registry = nullptr;
        };

        /**
         * @brief Lists the extents of every output a traced model published.
         * @param graph The graph @p outputs names values of; a value of another graph is fatal.
         * @return One entry per output in published order, or OutOfMemory.
         */
        [[nodiscard]] Result<std::vector<OutputShape>> list_output_shapes(
            const Graph::Graph& graph, const NN::ModelOutputs& outputs) noexcept;

        /**
         * @brief Returns the entry of @p outputs named @p name.
         * @return A pointer into @p outputs, or null when no output has that name.
         */
        [[nodiscard]] const OutputShape* find_output_shape(
            const std::vector<OutputShape>& outputs, std::string_view name) noexcept;

    } // namespace Train
} // namespace SushiAI
