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
 * @file sum_objective.cpp
 * @brief Implements the composite objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/train/sum_objective.hpp>

#include <cmath>
#include <utility>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/seam.hpp>
#include <SushiAI/core/shape.hpp>
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
                    SA_TRY_ASSIGN(term.weight,
                                  Config::read_real(entry.at(K_SUM_TERM_WEIGHT_KEY, named), named));
                }

                // The term's factory refuses a key it does not know, so it sees its own alone.
                const Config::Value own =
                    entry.without(K_SUM_TERM_TYPE_KEY).without(K_SUM_TERM_WEIGHT_KEY);
                SA_TRY_ASSIGN(term.objective,
                              make_objective(*given.registry, type->as_string(where), given, own,
                                             where));

                return Result<SumTerm>(std::move(term));
            }
        } // namespace

        Result<std::unique_ptr<SumObjective>> SumObjective::make(std::vector<SumTerm> terms)
        {
            SA_FAIL_IF(terms.empty(), StatusCode::InvalidDescription,
                       "A sum objective adds the losses of at least one term; it was given none");

            for (std::size_t index = 0; index < terms.size(); ++index)
            {
                const SumTerm& term = terms[index];
                SA_FATAL_IF(term.objective == nullptr, "Term {} of a sum objective is null",
                            index);
                SA_FAIL_IF(!std::isfinite(term.weight) || !(term.weight > 0.0),
                           StatusCode::InvalidDescription,
                           "Term {} of a sum objective, '{}', has the weight {}; a weight is "
                           "finite and above zero",
                           index, term.objective->name(), term.weight);
                SA_FAIL_IF(!term.objective->is_fed_by_batch(), StatusCode::InvalidDescription,
                           "Term {} of a sum objective is '{}', which is fed by its own step; "
                           "a sum is fed from a batch alone and takes no such term",
                           index, term.objective->name());
            }

            return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                return std::unique_ptr<SumObjective>(new SumObjective(std::move(terms)));
            });
        }

        SumObjective::SumObjective(std::vector<SumTerm> terms) noexcept
            : terms_(std::move(terms))
        {

        }

        std::string SumObjective::leaf_prefix(std::size_t index)
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

        Graph::ValueID SumObjective::loss(const Graph::GraphBuilder& builder,
                                          const Graph::Graph& graph,
                                          const NN::ModelOutputs& outputs,
                                          const ObjectiveLeaves& leaves) const
        {
            Graph::ValueID total = Graph::K_NO_VALUE;
            DType total_dtype = DType::Float32;
            for (std::size_t index = 0; index < terms_.size(); ++index)
            {
                const SumTerm& term = terms_[index];
                const ObjectiveLeaves own = leaves_of_term(leaves, index);
                Graph::ValueID value = term.objective->loss(builder, graph, outputs, own);

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
                    value = builder.reshape(value, Shape{1});

                // A weight of exactly 1 changes nothing, so it emits no node.
                if (term.weight != 1.0)
                    value = builder.scale(value, term.weight);

                total = total == Graph::K_NO_VALUE ? value : builder.add(total, value);
                total_dtype = dtype;
            }

            return total;
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

        SushiRuntime::span<const SumTerm> SumObjective::terms() const noexcept
        {
            return SushiRuntime::span<const SumTerm>(terms_.data(), terms_.size());
        }

        Result<std::unique_ptr<Objective>> make_sum_objective(const ObjectiveContext& given,
                                                              const Config::Value& attributes,
                                                              std::string_view context)
        {
            SA_FAIL_IF(given.registry == nullptr, StatusCode::InvalidDescription,
                       "{} is a sum of objectives, which a registry builds, and it was given "
                       "none; build it through make_objective",
                       context);

            const Config::SectionReader reader(attributes, context);
            SA_TRY(reader.check_keys(Config::KeySpan(&K_SUM_TERMS_KEY, 1)));
            SA_TRY_ASSIGN(const Config::Value* const listed,
                          reader.find_of(K_SUM_TERMS_KEY, Config::Kind::Array));

            const std::vector<Config::Value>& entries = listed->as_array(context);
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
