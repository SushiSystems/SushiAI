/**************************************************************************/
/* evaluation_step.hpp                                                    */
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
 * @file evaluation_step.hpp
 * @brief Declares EvaluationStep, the compiled forward-only replay of a traced graph.
 * @author Mustafa Garip
 */

#pragma once

#include <memory>
#include <string_view>
#include <SushiBLAS/engine.hpp>
#include <SushiBLAS/tensor.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>

namespace SushiAI
{
    namespace NN
    {
        class ModelState;
    } // namespace NN

    namespace Train
    {
        struct TracedStep;

        /**
         * @brief Owns one compiled forward-only replay of a traced graph.
         *
         * It serves a probe graph as well as an evaluation graph. The pool holds every value of
         * the graph, and the state leaves share their storage with the model state.
         */
        class EvaluationStep final
        {
            public:
                /**
                 * @brief Binds @p traced to @p state and @p pool, then compiles it once.
                 * @param engine Must outlive the step, as must @p state and @p pool.
                 * @param traced Read during the call only; the step copies its outputs.
                 * @param pool Holds no leaf of the graph yet.
                 * @return The step, InvalidDescription for a graph with no input leaf or one the
                 *         planner refuses, or the failure of the binding or the compilation.
                 */
                [[nodiscard]] static Result<std::unique_ptr<EvaluationStep>> make(
                    SushiBLAS::Engine& engine, const TracedStep& traced,
                    const NN::ModelState& state, Graph::TensorPool& pool,
                    const Graph::FusionSelector* selector = nullptr);

                /** @brief Deleted: a step holds one compiled plan and is not copied. */
                EvaluationStep(const EvaluationStep&) = delete;

                /** @brief Deleted: a step holds one compiled plan and is not copied. */
                EvaluationStep& operator=(const EvaluationStep&) = delete;

                /** @brief Deleted: the compiled plan stays where it was built. */
                EvaluationStep(EvaluationStep&&) = delete;

                /** @brief Deleted: the compiled plan stays where it was built. */
                EvaluationStep& operator=(EvaluationStep&&) = delete;

                /** @brief Releases the compiled step; the pool and the state are not owned. */
                ~EvaluationStep();

                /**
                 * @brief Copies @p inputs into the input leaf and replays the plan.
                 * @param inputs A tensor with the rank, extents and dtype of the input leaf.
                 * @pre The engine holds no recorded task; the copy runs everything recorded.
                 * @return Success, Incompatible from check_feed(), or the failure of the copy or
                 *         of the replay.
                 */
                [[nodiscard]] Status run(const SushiBLAS::Tensor& inputs);

                /**
                 * @brief Returns the tensor behind the output published under @p name.
                 * @return A tensor the pool owns, holding what the last run() computed, or
                 *         InvalidDescription listing the names that exist.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> output(std::string_view name) const;

                /** @brief Returns the compiled step, for its counters. */
                [[nodiscard]] const Graph::CompiledStep& step() const noexcept;

            private:
                /**
                 * @brief Takes the parts make() built.
                 * @param input The input leaf's tensor in @p pool.
                 */
                EvaluationStep(SushiBLAS::Engine& engine, Graph::TensorPool& pool,
                               SushiBLAS::Tensor& input, NN::ModelOutputs outputs,
                               std::unique_ptr<Graph::CompiledStep> step) noexcept;

                /** @brief The engine that records and runs the copy of each feed. */
                SushiBLAS::Engine* engine_;

                /** @brief The pool that holds every value of the graph; not owned. */
                Graph::TensorPool* pool_;

                /** @brief The input leaf's tensor, which the pool owns. */
                SushiBLAS::Tensor* input_;

                /** @brief The values the model published, by name. */
                NN::ModelOutputs outputs_;

                /** @brief The plan compiled from the traced graph. */
                std::unique_ptr<Graph::CompiledStep> step_;
        };

    } // namespace Train
} // namespace SushiAI
