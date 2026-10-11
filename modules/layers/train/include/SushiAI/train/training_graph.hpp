/**************************************************************************/
/* training_graph.hpp                                                     */
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
 * @file training_graph.hpp
 * @brief Declares TrainingGraph, the traced and differentiated graph of one training step.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/train/objective.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @struct StepSpec
         * @brief The four graph values a training loop has to know by name.
         */
        struct StepSpec
        {
            /** @brief The [N, D] minibatch input leaf, rewritten per step. */
            Graph::ValueID inputs = Graph::K_NO_VALUE;

            /** @brief The [N, C] one-hot target leaf, rewritten per step. */
            Graph::ValueID targets = Graph::K_NO_VALUE;

            /** @brief The model's [N, C] logits, read for accuracy. */
            Graph::ValueID logits = Graph::K_NO_VALUE;

            /** @brief The 1-element loss that was differentiated. */
            Graph::ValueID loss = Graph::K_NO_VALUE;
        };

        /**
         * @struct ParameterBinding
         * @brief A parameter's name and the two graph values behind it.
         */
        struct ParameterBinding
        {
            /** @brief The dotted name from the parameter walk, e.g. "0.weight". */
            std::string name;

            /** @brief The parameter's value id. */
            Graph::ValueID value = Graph::K_NO_VALUE;

            /** @brief Its gradient's value id; Graph::K_NO_VALUE if none flows. */
            Graph::ValueID gradient = Graph::K_NO_VALUE;

            /** @brief What the parameter is for, which decides whether weight decay applies. */
            NN::ParameterRole role = NN::ParameterRole::Weight;
        };

        /**
         * @struct TrainingGraph
         * @brief A traced, differentiated model, ready to lower.
         */
        struct TrainingGraph
        {
            /** @brief The forward and backward IR, in one graph. */
            Graph::Graph graph;

            /** @brief The values the loop reads and writes. */
            StepSpec spec;

            /** @brief Every parameter, paired with its gradient. */
            std::vector<ParameterBinding> parameters;

            /**
             * @brief The 1-element leaf the backward seed is multiplied by.
             */
            Graph::ValueID loss_scale = Graph::K_NO_VALUE;

            /**
             * @brief Holds leaves declared by the objective, indexed by role.
             */
            ObjectiveLeaves leaves;

            /** @brief What the cast pass converted, and where. */
            Graph::CastReport casts;

            /** @brief What the fusion pass did to @c graph, and what it refused. */
            Graph::FusionReport fusion;

            /** @brief Holds every value the model published, by name. */
            NN::ModelOutputs outputs;

            /** @brief Holds the buffer updates the training trace requested. */
            NN::BufferUpdateList updates;

            /** @brief Lists the leaves the step fills once with numbers of the host. */
            NN::HostTableList tables;

            /** @brief Lists the values the memory planner must not pack. */
            std::vector<Graph::ValueID> pinned;
        };

    } // namespace Train
} // namespace SushiAI
