/**************************************************************************/
/* objective.hpp                                                          */
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
 * @file objective.hpp
 * @brief Defines training objectives and their metrics.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiBLAS/engine/engine.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/step_view.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @class ObjectiveLeaves
         * @brief Holds the graph leaves an objective declared and the batch roles it reads.
         *
         * For cross-entropy that is one [N, C] target leaf, built from the labels role of
         * a batch; for a PINN it would be the collocation points.
         */
        class ObjectiveLeaves
        {
            public:
                /**
                 * @brief Records that @p role is @p id.
                 * @param role What the leaf is for, e.g. "targets".
                 * @param id The declared value.
                 * @throws Error If @p role was already declared.
                 */
                void add(std::string_view role, Graph::ValueID id);

                /**
                 * @brief Whether @p role was declared.
                 * @param role What the leaf is for.
                 * @return True when it was.
                 */
                bool has(std::string_view role) const noexcept;

                /**
                 * @brief The value declared for @p role.
                 * @param role What the leaf is for.
                 * @return Its value id.
                 * @throws Error If @p role was not declared; lists what was.
                 */
                Graph::ValueID at(std::string_view role) const;

                /** @brief Every declared leaf, in declaration order. */
                const std::vector<std::pair<std::string, Graph::ValueID>>& all() const noexcept
                {
                    return leaves_;
                }

                /**
                 * @brief Records that the objective reads @p role from every batch.
                 * @param role The role, its per-sample shape and its element type; a role
                 *             may be required more than once, and each entry is matched.
                 */
                void require(Data::TargetDescription role);

                /** @brief Returns every batch role the objective reads, in declaration order. */
                [[nodiscard]] const std::vector<Data::TargetDescription>& required() const noexcept
                {
                    return required_;
                }

            private:
                /** @brief Holds every declared leaf with its role, in declaration order. */
                std::vector<std::pair<std::string, Graph::ValueID>> leaves_;

                /** @brief Holds every batch role the objective reads, in declaration order. */
                std::vector<Data::TargetDescription> required_;
        };

        /**
         * @brief Returns the tensor @p pool holds for the leaf @p leaves declared as @p role.
         * @return The tensor, never null, or InvalidDescription when @p leaves declared no
         *         such role or @p pool holds no tensor for it.
         */
        [[nodiscard]] Result<SushiBLAS::Tensor*> find_leaf_tensor(const ObjectiveLeaves& leaves,
                                                                  std::string_view role,
                                                                  Graph::TensorPool& pool);

        /**
         * @class Objective
         * @brief What a model is being trained to minimise.
         */
        class Objective
        {
            public:
                virtual ~Objective() = default;

                /**
                 * @brief Declares the leaves the graph needs beyond the model's input, and
                 *        the batch roles they are filled from.
                 * @param builder Where the leaves go.
                 * @param leaves Where to record them, by role.
                 */
                virtual void declare(const Graph::GraphBuilder& builder,
                                     ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief Computes the scalar loss from the model's named outputs.
                 * @param builder Where the nodes go.
                 * @param graph The graph, for checking shapes.
                 * @param outputs The model's outputs; the objective names the ones it reads.
                 * @param leaves Declared auxiliary leaves.
                 * @return The scalar loss value identifier.
                 */
                virtual Graph::ValueID loss(const Graph::GraphBuilder& builder,
                                            const Graph::Graph& graph,
                                            const NN::ModelOutputs& outputs,
                                            const ObjectiveLeaves& leaves) const = 0;

                /**
                 * @brief Writes the leaves declare() recorded from the roles of @p batch.
                 * @param leaves The leaves this objective declared.
                 * @param pool Holds the tensor of every leaf; no recorded task reads them.
                 * @return Success; Unsupported, the default, for an objective whose targets
                 *         its own step computes; or the refusal of a role or a value.
                 */
                [[nodiscard]] virtual Status feed(const Data::Batch& batch,
                                                  const ObjectiveLeaves& leaves,
                                                  Graph::TensorPool& pool) const;

                /**
                 * @brief Reports whether feed() fills the leaves from a batch alone.
                 * @return False, the default, for an objective whose own step computes its
                 *         targets; an objective that overrides feed() overrides this too.
                 */
                [[nodiscard]] virtual bool is_fed_by_batch() const noexcept;

                /**
                 * @brief Appends the named numbers of a step that has run to @p metrics.
                 * @param step The batch, the loss and the outputs the step left on the host.
                 * @return Success, having appended nothing by default, or the refusal of
                 *         an output or a role the metric reads.
                 */
                [[nodiscard]] virtual Status measure(const StepView& step,
                                                     Metrics& metrics) const;

                /** @brief A short name for a run report. */
                virtual std::string_view name() const noexcept = 0;
        };

        /**
         * @class CrossEntropyObjective
         * @brief Multi-class cross-entropy over one-hot targets, and accuracy.
         *
         * Emits @c nll_loss(log_softmax(output), targets). The targets leaf holds one-hot
         * rows, which feed() writes from the labels role of each batch.
         */
        class CrossEntropyObjective final : public Objective
        {
            public:
                /**
                 * @brief Constructs the objective for a fixed batch and class count.
                 * @param batch Samples per step.
                 * @param classes How many classes.
                 * @param dtype The element type the target leaf carries.
                 */
                CrossEntropyObjective(int64_t batch, int64_t classes,
                                      DType dtype = DType::Float32) noexcept
                    : batch_(batch), classes_(classes), dtype_(dtype) {}

                void declare(const Graph::GraphBuilder& builder,
                             ObjectiveLeaves& leaves) const override;

                Graph::ValueID loss(const Graph::GraphBuilder& builder,
                                    const Graph::Graph& graph,
                                    const NN::ModelOutputs& outputs,
                                    const ObjectiveLeaves& leaves) const override;

                /** @brief Writes the one-hot row of every label with fill_one_hot(). */
                [[nodiscard]] Status feed(const Data::Batch& batch,
                                          const ObjectiveLeaves& leaves,
                                          Graph::TensorPool& pool) const override;

                /** @brief Returns true: a batch alone fills the leaf. */
                [[nodiscard]] bool is_fed_by_batch() const noexcept override { return true; }

                /**
                 * @brief Appends "accuracy": the fraction of rows whose label is the
                 *        largest logit.
                 * @return InvalidDescription for a step of several outputs, logits the
                 *         host cannot read or a batch with no labels role; Incompatible
                 *         for another count of logits or labels.
                 */
                [[nodiscard]] Status measure(const StepView& step,
                                             Metrics& metrics) const override;

                std::string_view name() const noexcept override { return "cross_entropy"; }

            private:
                int64_t batch_ = 0;
                int64_t classes_ = 0;
                DType dtype_ = DType::Float32;
        };

        /**
         * @brief Builds a cross-entropy objective over the batch and classes of @p given.
         * @param attributes The objective's own keys; it has none.
         * @param context What to call this objective in a diagnostic.
         * @return The objective, or InvalidDescription for a key, or for a batch or a class
         *         count below 1.
         */
        [[nodiscard]] Result<std::unique_ptr<Objective>> make_cross_entropy(
            const ObjectiveContext& given, const Config::Value& attributes,
            std::string_view context);

    } // namespace Train
} // namespace SushiAI
