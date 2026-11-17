/**************************************************************************/
/* sum_objective.cpp                                                      */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file sum_objective.cpp
 * @brief Implements the composite objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/sum_objective.hpp>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_registry.hpp>

namespace SushiAI
{
    namespace Train
    {
        namespace
        {
            /** @brief Returns the leaves @p all holds for term @p index, by the term's roles. */
            [[nodiscard]] ObjectiveLeaves leaves_of_term(const ObjectiveLeaves& all,
                                                         std::size_t index)
            {
                const std::string prefix = SumObjective::leaf_prefix(index);
                ObjectiveLeaves own;
                for (const std::pair<std::string, Graph::ValueID>& leaf : all.all())
                {
                    if (leaf.first.compare(0, prefix.size(), prefix) == 0)
                        own.add(std::string_view(leaf.first).substr(prefix.size()), leaf.second);
                }

                return own;
            }

            /**
             * @brief Reads one entry of "terms", named @p where, and builds its objective.
             * @pre @p given holds a registry.
             * @return The term, or InvalidDescription for an entry that is no object, has no
             *         string "type" or a "weight" that is no number, or its factory's refusal.
             */
            [[nodiscard]] Result<SumTerm> read_term(const ObjectiveContext& given,
                                                    const Config::Value& entry,
                                                    std::string_view where)
            {
                const Config::SectionReader reader(entry, where);
                SA_TRY(reader.check_object());
                SA_TRY_ASSIGN(const Config::Value* const type,
                              reader.find_of(K_SUM_TERM_TYPE_KEY, Config::Kind::String));

                SumTerm term;
                if (reader.has(K_SUM_TERM_WEIGHT_KEY))
                {
                    const std::string named = reader.member(K_SUM_TERM_WEIGHT_KEY);
                    SA_TRY_ASSIGN(const Config::Value* const weight,
                                  reader.find(K_SUM_TERM_WEIGHT_KEY));
                    SA_TRY_ASSIGN(term.weight, Config::read_real(*weight, named));
                }

                // The term's factory refuses a key it does not know, so it sees its own alone.
                const Config::Value own =
                    entry.without(K_SUM_TERM_TYPE_KEY).without(K_SUM_TERM_WEIGHT_KEY);
                SA_TRY_ASSIGN(const std::string& type_name, type->as_string(where));
                SA_TRY_ASSIGN(term.objective,
                              make_objective(*given.registry, type_name, given, own, where));

                return Result<SumTerm>(std::move(term));
            }

            /**
             * @brief Returns the metric names of every term of @p terms, each once, in order.
             * @return Empty when a term states none: the sum cannot state what it appends then.
             */
            [[nodiscard]] std::vector<std::string_view> merged_names(
                const std::vector<SumTerm>& terms)
            {
                std::vector<std::string_view> merged;
                for (const SumTerm& term : terms)
                {
                    const SushiRuntime::span<const std::string_view> stated =
                        term.objective->metric_names();
                    if (stated.empty())
                        return {};

                    for (const std::string_view name : stated)
                    {
                        if (std::find(merged.begin(), merged.end(), name) == merged.end())
                            merged.push_back(name);
                    }
                }

                return merged;
            }

            /** @brief Reports whether @p metrics holds every name of @p names. */
            [[nodiscard]] bool holds_every(const Metrics& metrics,
                                           SushiRuntime::span<const std::string_view> names) noexcept
            {
                for (const std::string_view name : names)
                {
                    if (metrics.find(name) == nullptr)
                        return false;
                }

                return !names.empty();
            }
        } // namespace

        Result<std::unique_ptr<SumObjective>> SumObjective::make(std::vector<SumTerm> terms) noexcept
        {
            SA_FAIL_IF(terms.empty(), StatusCode::InvalidDescription,
                       "A sum objective adds the losses of at least one term; it was given none");

            for (std::size_t index = 0; index < terms.size(); ++index)
            {
                const SumTerm& term = terms[index];
                SA_FATAL_IF(term.objective == nullptr, "Term {} of a sum objective is null",
                            index);
                SA_FAIL_IF(!std::isfinite(term.weight) || !(term.weight >= 0.0),
                           StatusCode::InvalidDescription,
                           "Term {} of a sum objective, '{}', has the weight {}; a weight is "
                           "finite and at least zero",
                           index, term.objective->name(), term.weight);
                SA_FAIL_IF(!term.objective->is_fed_by_batch(), StatusCode::InvalidDescription,
                           "Term {} of a sum objective is '{}', which is fed by its own step; "
                           "a sum is fed from a batch alone and takes no such term",
                           index, term.objective->name());
            }

            const bool has_weight = std::any_of(terms.begin(), terms.end(),
                                                [](const SumTerm& term)
            {
                return term.weight > 0.0;
            });
            SA_FAIL_IF(!has_weight, StatusCode::InvalidDescription,
                       "Every term of a sum objective has the weight 0, so its loss is 0 and no "
                       "parameter has a gradient; give one term a weight above zero");

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::vector<std::string_view> names = merged_names(terms);

                return std::unique_ptr<SumObjective>(
                    new SumObjective(std::move(terms), std::move(names)));
            });
        }

        SumObjective::SumObjective(std::vector<SumTerm> terms,
                                   std::vector<std::string_view> names) noexcept
            : terms_(std::move(terms)), names_(std::move(names))
        {

        }

        std::string SumObjective::leaf_prefix(std::size_t index) noexcept
        {
            return "term" + std::to_string(index) + ".";
        }

        void SumObjective::declare(const Graph::GraphBuilder& builder,
                                   ObjectiveLeaves& leaves) const
        {
            for (std::size_t index = 0; index < terms_.size(); ++index)
            {
                ObjectiveLeaves own;
                terms_[index].objective->declare(builder, own);

                const std::string prefix = leaf_prefix(index);
                for (const std::pair<std::string, Graph::ValueID>& leaf : own.all())
                    leaves.add(prefix + leaf.first, leaf.second);

                for (const Data::TargetDescription& role : own.required())
                    leaves.require(role);
            }
        }

        Result<LossTrace> SumObjective::loss(
            const Graph::GraphBuilder& builder, const Graph::Graph& graph,
            const NN::ModelOutputs& outputs, const ObjectiveLeaves& leaves) const
        {
            Graph::ValueID total = Graph::K_NO_VALUE;
            std::vector<LossTrace> traces;
            traces.reserve(terms_.size());
            DType total_dtype = DType::Float32;
            for (std::size_t index = 0; index < terms_.size(); ++index)
            {
                const SumTerm& term = terms_[index];
                const ObjectiveLeaves own = leaves_of_term(leaves, index);
                SA_TRY_ASSIGN(LossTrace term_trace,
                              term.objective->loss(builder, graph, outputs, own));
                Graph::ValueID value = term_trace.loss();
                traces.push_back(std::move(term_trace));

                // By value: every emission below can reallocate the value vector.
                const Shape shape = graph.value(value).shape;
                const DType dtype = graph.value(value).dtype;
                SA_FATAL_IF(shape.num_elements() != 1,
                            "Term {} of a sum objective, '{}', computed a loss of {}; a loss "
                            "is one element",
                            index, term.objective->name(), shape.to_string());
                SA_FATAL_IF(total != Graph::K_NO_VALUE && dtype != total_dtype,
                            "Term {} of a sum objective, '{}', computed a {} loss, but the "
                            "terms before it computed {}",
                            index, term.objective->name(), dtype_name(dtype),
                            dtype_name(total_dtype));

                if (shape != Shape{1})
                {
                    SA_TRY_ASSIGN(value, builder.reshape(value, Shape{1}));
                }

                // A weight of exactly 1 changes nothing, so it emits no node.
                if (term.weight != 1.0)
                {
                    SA_TRY_ASSIGN(value, builder.scale(value, term.weight));
                }

                if (total == Graph::K_NO_VALUE)
                {
                    total = value;
                }
                else
                {
                    SA_TRY_ASSIGN(total, builder.add(total, value));
                }
                total_dtype = dtype;
            }

            LossTrace trace(total);
            for (LossTrace& term_trace : traces)
                trace.add_term(std::move(term_trace));

            return trace;
        }

        Result<NN::ModelOutputs> SumObjective::declare_metric_values(
            const Graph::GraphBuilder& builder, const NN::ModelOutputs& outputs,
            const LossTrace& trace) const
        {
            NN::ModelOutputs merged;
            for (std::size_t index = 0; index < terms_.size(); ++index)
            {
                SA_TRY_ASSIGN(const NN::ModelOutputs own,
                              terms_[index].objective->declare_metric_values(
                                  builder, outputs, trace.term(index)));
                for (const NN::ModelOutputs::Entry& entry : own.all())
                {
                    if (merged.at(entry.first).has_value())
                        continue;

                    SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate,
                                                  [&]() -> Status
                    {
                        return merged.add(entry.first, entry.second);
                    }));
                }
            }

            return Result<NN::ModelOutputs>(std::move(merged));
        }

        Status SumObjective::feed(const Data::Batch& batch, const ObjectiveLeaves& leaves,
                                  Graph::TensorPool& pool) const
        {
            for (std::size_t index = 0; index < terms_.size(); ++index)
            {
                SA_TRY_ASSIGN(const ObjectiveLeaves own,
                              SushiAI::Detail::guard_value(
                                  SushiAI::Detail::SeamPhase::Allocate, [&leaves, index]()
                {
                    return leaves_of_term(leaves, index);
                }));
                SA_TRY(terms_[index].objective->feed(batch, own, pool));
            }

            return Status();
        }

        Status SumObjective::measure(const StepView& step, Metrics& metrics) const
        {
            for (const SumTerm& term : terms_)
            {
                // A term whose metrics are all appended already would only compute them again.
                if (holds_every(metrics, term.objective->metric_names()))
                    continue;

                Metrics own;
                SA_TRY(term.objective->measure(step, own));

                for (const Metric& metric : own.all())
                {
                    // Of two metrics of one name the first term's is the one reported.
                    if (metrics.find(metric.name) == nullptr)
                        SA_TRY(metrics.add(metric));
                }
            }

            return Status();
        }

        SushiRuntime::span<const std::string_view> SumObjective::metric_names() const noexcept
        {
            return SushiRuntime::span<const std::string_view>(names_.data(), names_.size());
        }

        SushiRuntime::span<const SumTerm> SumObjective::terms() const noexcept
        {
            return SushiRuntime::span<const SumTerm>(terms_.data(), terms_.size());
        }

        Result<std::unique_ptr<Objective>> make_sum_objective(const ObjectiveContext& given,
                                                              const Config::Value& attributes,
                                                              std::string_view context) noexcept
        {
            SA_FAIL_IF(given.registry == nullptr, StatusCode::InvalidDescription,
                       "{} is a sum of objectives, which a registry builds, and it was given "
                       "none; build it through make_objective",
                       context);

            const Config::SectionReader reader(attributes, context);
            SA_TRY(reader.check_keys(Config::KeySpan(&K_SUM_TERMS_KEY, 1)));
            SA_TRY_ASSIGN(const Config::Value* const listed,
                          reader.find_of(K_SUM_TERMS_KEY, Config::Kind::Array));

            SA_TRY_ASSIGN(const std::vector<Config::Value>& entries, listed->as_array(context));
            const std::string where = reader.member(K_SUM_TERMS_KEY);
            std::vector<SumTerm> terms;
            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                terms.reserve(entries.size());
                for (std::size_t index = 0; index < entries.size(); ++index)
                {
                    SA_TRY_ASSIGN(SumTerm term, read_term(given, entries[index],
                                                          Config::entry_of(where, index)));
                    terms.push_back(std::move(term));
                }

                return Status();
            }));

            SA_TRY_ASSIGN(std::unique_ptr<SumObjective> made,
                          SumObjective::make(std::move(terms)));
            std::unique_ptr<Objective> base = std::move(made);

            return std::move(base);
        }

    } // namespace Train
} // namespace SushiAI
