/**************************************************************************/
/* classifier_objective.hpp                                               */
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
 * @file classifier_objective.hpp
 * @brief Defines the objective of a graph that Train::build_classifier() traced.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/graph/ir.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/training_graph.hpp>

namespace SushiAI
{
    namespace Test
    {
        /**
         * @brief Returns the cross-entropy objective build_classifier() traced @p built with.
         * @pre @p built holds the [rows, classes] target leaf build_classifier() declares.
         * @return An objective a Trainer over @p built is made with; it outlives the trainer.
         */
        [[nodiscard]] inline Train::CrossEntropyObjective classifier_objective(
            const Train::TrainingGraph& built) noexcept
        {
            const Graph::Value& targets = built.graph.value(built.spec.targets);

            return Train::CrossEntropyObjective(targets.shape[0], targets.shape[1], targets.dtype);
        }

    } // namespace Test
} // namespace SushiAI
