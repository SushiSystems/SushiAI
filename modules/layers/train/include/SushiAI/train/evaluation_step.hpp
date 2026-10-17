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
#include <string>
#include <string_view>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>

namespace SushiAI
{
    namespace NN
    {
        class ModelState;
    } // namespace NN

    namespace Train
    {
        struct TracedStep;

        /** @brief Pairs one tensor of a run with the name of the leaf it is copied into. */
        struct NamedFeed
        {
            /** @brief Names an input leaf of the step. */
            std::string_view name;

            /** @brief Points at the tensor to copy; never null. */
            const SushiBLAS::Tensor* tensor = nullptr;
        };

        /** @brief Views the tensors of a run, one per input leaf, in the step's order. */
        using FeedSpan = SushiRuntime::span<const SushiBLAS::Tensor* const>;

        /**
         * @brief Owns one compiled forward-only replay of a traced graph.
         *
         * It serves a probe graph, an evaluation graph and a carrying graph. The pool holds
         * every value of the graph, and the state leaves share their storage with the model
         * state. A carrying graph adds leaves the caller feeds and leaves the step rewrites.
         */
        class EvaluationStep final
        {
            public:
                /**
                 * @brief Binds @p traced to @p state and @p pool, then compiles it once.
                 * @param engine Must outlive the step, as must @p state and @p pool.
                 * @param traced Read during the call only; the step copies what it keeps.
                 * @param pool Holds no leaf of the graph yet; borrows the arenas of @p lender.
                 * @return The step, InvalidDescription for a graph with no input leaf, a fed
                 *         leaf named as the model input is, or a plan the planner refuses.
                 */
                [[nodiscard]] static Result<std::unique_ptr<EvaluationStep>> make(
                    SushiBLAS::Engine& engine, const TracedStep& traced,
                    const NN::ModelState& state, Graph::TensorPool& pool,
                    const Graph::FusionSelector* selector = nullptr,
                    const Graph::TensorPool* lender = nullptr);

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
                 * @brief Copies @p inputs into the model input and replays the plan.
                 * @param inputs A tensor with the rank, extents and dtype of the input leaf.
                 * @pre The engine holds no recorded task; the copy runs everything recorded.
                 * @return Success, Incompatible from check_feed() or for a step that is fed
                 *         further leaves, or the failure of the copy or of the replay.
                 */
                [[nodiscard]] Status run(const SushiBLAS::Tensor& inputs);

                /**
                 * @brief Copies one tensor into each input leaf, then replays the plan.
                 * @param feeds One tensor per entry of input_names(), in that order; a null
                 *              entry is fatal.
                 * @pre The engine holds no recorded task; the copies run everything recorded.
                 * @return Success, Incompatible for another count or from check_feed(), or the
                 *         failure of a copy or of the replay.
                 */
                [[nodiscard]] Status run_in_order(FeedSpan feeds);

                /**
                 * @brief Copies each tensor into the leaf it names, then replays the plan.
                 * @param feeds One entry per input leaf, in any order.
                 * @return What run_in_order() returns, or InvalidDescription for a name that
                 *         is no input leaf, is given twice or is left out.
                 */
                [[nodiscard]] Status run_by_name(SushiRuntime::span<const NamedFeed> feeds);

                /**
                 * @brief Fills every carried leaf with its initial value.
                 * @pre The engine holds no recorded task; the fills run everything recorded.
                 * @return Success, at once for a step that carries nothing, or the failure of
                 *         a fill.
                 */
                [[nodiscard]] Status reset();

                /** @brief Reports whether the step keeps a value from a run to the next. */
                [[nodiscard]] bool carries_state() const noexcept;

                /**
                 * @brief Lists the input leaves: the model input, then the fed leaves in the
                 *        order their trace declared them.
                 */
                [[nodiscard]] SushiRuntime::span<const std::string> input_names() const noexcept;

                /**
                 * @brief Returns the leaf a run copies the feed named @p name into.
                 * @return A tensor the pool owns, or InvalidDescription listing the names
                 *         that exist.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> input(std::string_view name) const;

                /**
                 * @brief Returns the tensor behind the output published under @p name.
                 * @return A tensor the pool owns, holding what the last run computed, or
                 *         InvalidDescription listing the names that exist.
                 */
                [[nodiscard]] Result<const SushiBLAS::Tensor*> output(std::string_view name) const;

                /** @brief Returns the compiled step, for its counters. */
                [[nodiscard]] const Graph::CompiledStep& step() const noexcept;

                /**
                 * @brief Sets the run count of the compiled step to that of a saved run.
                 * @return What Graph::CompiledStep::restore_run_count() returns.
                 */
                [[nodiscard]] Status restore_run_count(std::size_t count);

            private:
                /** @brief Holds one carried leaf and what reset() fills it with. */
                struct CarriedLeaf
                {
                    SushiBLAS::Tensor* tensor = nullptr; /**< The leaf, which the pool owns. */
                    double initial_value = 0.0;          /**< What reset() writes. */
                };

                /** @brief Holds what make() gathers before the plan is compiled. */
                struct Parts
                {
                    std::vector<std::string> input_names;        /**< One per input leaf. */
                    std::vector<SushiBLAS::Tensor*> input_leaves; /**< In the same order. */
                    std::vector<CarriedLeaf> carried;             /**< One per carried value. */
                    std::vector<BufferUpdateSlot> carries;        /**< leaf = next, per value. */
                    NN::ModelOutputs outputs;                     /**< What the model published. */
                };

                /** @brief Takes the parts make() built; the plan is compiled afterwards. */
                EvaluationStep(SushiBLAS::Engine& engine, Graph::TensorPool& pool, Parts parts);

                /**
                 * @brief Gathers the leaves, the carried pairs and the outputs of @p traced.
                 * @pre Every leaf of the graph and the next value of every carried pair is
                 *      materialised in @p pool.
                 */
                [[nodiscard]] static Result<Parts> gather(const TracedStep& traced,
                                                          Graph::TensorPool& pool);

                /** @brief Joins the input names with ", " for a diagnostic. */
                [[nodiscard]] std::string list_inputs() const;

                /** @brief The engine that records and runs the copy of each feed. */
                SushiBLAS::Engine* engine_;

                /** @brief The pool that holds every value of the graph; not owned. */
                Graph::TensorPool* pool_;

                /** @brief The name of each input leaf, the model input first. */
                std::vector<std::string> input_names_;

                /** @brief The tensor of each input leaf, in the order of input_names_. */
                std::vector<SushiBLAS::Tensor*> input_leaves_;

                /** @brief The feeds of run_by_name() in the step's order, kept between runs. */
                std::vector<const SushiBLAS::Tensor*> ordered_;

                /** @brief The leaves the step rewrites after each run. */
                std::vector<CarriedLeaf> carried_;

                /** @brief The values the model published, by name. */
                NN::ModelOutputs outputs_;

                /** @brief Records leaf = next for every carried value behind the graph. */
                BufferUpdateRecorder carries_;

                /** @brief The plan compiled from the traced graph; declared last, dies first. */
                std::unique_ptr<Graph::CompiledStep> step_;
        };

    } // namespace Train
} // namespace SushiAI
