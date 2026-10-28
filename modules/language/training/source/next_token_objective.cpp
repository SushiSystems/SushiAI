/**************************************************************************/
/* next_token_objective.cpp                                               */
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
 * @file next_token_objective.cpp
 * @brief Implements the next-token objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/next_token_objective.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/train/batch_role.hpp>

namespace SushiAI::Language::Training
{
    namespace
    {
        /** @brief Names the rank of the [N, T, V] logits the objective reads. */
        constexpr std::int32_t K_LOGITS_RANK = 3;

        /** @brief Holds the largest index K_GATHER_ROWS addresses a table row by. */
        constexpr std::int64_t K_LARGEST_INDEX = std::numeric_limits<std::int32_t>::max();

        /**
         * @brief Reports whether INT32 indexes batch * length * vocabulary elements.
         * @pre Every argument is at least 1.
         */
        [[nodiscard]] bool fits_index(std::int64_t batch, std::int64_t length,
                                      std::int64_t vocabulary) noexcept
        {
            // Dividing first keeps each product below what int64 holds.
            return batch <= K_LARGEST_INDEX / length
                && batch * length <= K_LARGEST_INDEX / vocabulary;
        }

        /**
         * @brief Reads the name under K_NEXT_TOKEN_OUTPUT_KEY, or none when it is absent.
         * @return The name, empty for an absent key, or InvalidDescription for another
         *         key, or for a name that is no string or is empty.
         */
        [[nodiscard]] Result<std::string> read_output_name(const Config::Value& attributes,
                                                           std::string_view context)
        {
            for (const std::string& key : attributes.keys())
            {
                SA_FAIL_IF(key != K_NEXT_TOKEN_OUTPUT_KEY, StatusCode::InvalidDescription,
                           "{} states the key '{}', but a next_token objective reads the "
                           "key '{}' alone",
                           context, key, K_NEXT_TOKEN_OUTPUT_KEY);
            }

            if (!attributes.has(K_NEXT_TOKEN_OUTPUT_KEY))
                return std::string();

            SA_TRY_ASSIGN(const Config::Value& stated,
                          attributes.at(K_NEXT_TOKEN_OUTPUT_KEY, context));
            SA_FAIL_IF(stated.kind() != Config::Kind::String, StatusCode::InvalidDescription,
                       "{} '{}' names a published output, but it is {}", context,
                       K_NEXT_TOKEN_OUTPUT_KEY, Config::kind_name(stated.kind()));

            SA_TRY_ASSIGN(const std::string& name, stated.as_string(context));
            SA_FAIL_IF(name.empty(), StatusCode::InvalidDescription,
                       "{} '{}' names a published output, but it is empty", context,
                       K_NEXT_TOKEN_OUTPUT_KEY);

            return name;
        }

        /**
         * @brief Finds the output the objective reads: the one named, or the only one.
         * @return A pointer into @p given, never null, or InvalidDescription for a name no
         *         output has, or for no name and another count of outputs than one.
         */
        [[nodiscard]] Result<const Train::OutputShape*> find_logits(
            const Train::ObjectiveContext& given, const std::string& name,
            std::string_view context)
        {
            if (name.empty())
            {
                SA_FAIL_IF(given.outputs.size() != 1, StatusCode::InvalidDescription,
                           "{} reads one [N, T, V] output, but the model publishes {}; name "
                           "the logits with '{}'",
                           context, given.outputs.size(), K_NEXT_TOKEN_OUTPUT_KEY);

                return &given.outputs.front();
            }

            const Train::OutputShape* const named =
                Train::find_output_shape(given.outputs, name);
            SA_FAIL_IF(named == nullptr, StatusCode::InvalidDescription,
                       "{} reads its logits from the output '{}', but the model publishes "
                       "none of that name among its {}",
                       context, name, given.outputs.size());

            return named;
        }
    } // namespace

    Result<std::unique_ptr<NextTokenObjective>> NextTokenObjective::make(
        std::int64_t batch, std::int64_t length, std::int64_t vocabulary, std::string output) noexcept
    {
        SA_FAIL_IF(batch < 1 || length < 1 || vocabulary < 1, StatusCode::InvalidDescription,
                   "A next-token objective needs a batch, a length and a vocabulary of at "
                   "least 1; got {}, {} and {}",
                   batch, length, vocabulary);

        SA_FAIL_IF(!fits_index(batch, length, vocabulary), StatusCode::InvalidDescription,
                   "A next-token objective picks one of batch * length * vocabulary "
                   "log-probabilities by an int32 index; {} * {} * {} is above {}",
                   batch, length, vocabulary, K_LARGEST_INDEX);

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<NextTokenObjective>(
                new NextTokenObjective(batch, length, vocabulary, std::move(output)));
        });
    }

    NextTokenObjective::NextTokenObjective(std::int64_t batch, std::int64_t length,
                                           std::int64_t vocabulary,
                                           std::string output) noexcept
        : batch_(batch), length_(length), vocabulary_(vocabulary), output_(std::move(output))
    {

    }

    void NextTokenObjective::declare(const Graph::GraphBuilder& builder,
                                     Train::ObjectiveLeaves& leaves) const
    {
        leaves.add(K_NEXT_TOKEN_TARGETS_LEAF,
                   builder.input(Shape{batch_ * length_},
                                 K_NEXT_TOKEN_TARGETS_LEAF, DType::Int32));

        // No class count is stated: a file of a smaller vocabulary serves the same step.
        leaves.require(Data::TargetDescription{std::string(Data::K_ROLE_LABELS),
                                               Shape{length_}, DType::Int32, 0});
    }

    Result<Graph::ValueID> NextTokenObjective::loss(
        const Graph::GraphBuilder& builder, const Graph::Graph& graph,
        const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves& leaves) const
    {
        const Result<Graph::ValueID> found =
            output_.empty() ? outputs.only() : outputs.at(output_);
        SA_FATAL_IF(!found.has_value(), "The next-token objective reads one output: {}",
                    found.status().message());
        const Graph::ValueID logits = found.value();

        // By value: every emission below can reallocate the value vector.
        const Shape logits_shape = graph.value(logits).shape;
        const Shape expected{batch_, length_, vocabulary_};
        SA_FATAL_IF(logits_shape != expected,
                    "The model produced {} but the next-token objective was made for {}",
                    logits_shape.to_string(), expected.to_string());

        const std::int64_t rows = batch_ * length_;

        // LOG_SOFTMAX normalises each row, so every position becomes a row of V scores.
        const Graph::Traced log_probabilities =
            builder.log_softmax(builder.reshape(logits, Shape{rows, vocabulary_}));

        // As a table of rows * V single values, one index picks one log-probability.
        const Graph::Traced table =
            builder.reshape(log_probabilities, Shape{rows * vocabulary_, 1});
        const Graph::Traced picked =
            builder.gather_rows(table, leaves.at(K_NEXT_TOKEN_TARGETS_LEAF));
        SA_TRY_ASSIGN(const Graph::ValueID loss,
                      builder.scale(builder.sum_rows(picked),
                                    -1.0 / static_cast<double>(rows)));

        return loss;
    }

    Status NextTokenObjective::feed(const Data::Batch& batch,
                                    const Train::ObjectiveLeaves& leaves,
                                    Graph::TensorPool& pool) const
    {
        SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                      Train::require_batch_role(batch, Data::K_ROLE_LABELS));
        SA_TRY_ASSIGN(SushiBLAS::Tensor* const targets,
                      Train::find_leaf_tensor(leaves, K_NEXT_TOKEN_TARGETS_LEAF, pool));

        return fill_next_token_targets(labels->values, vocabulary_, *targets);
    }

    Status NextTokenObjective::measure(const Train::StepView& step,
                                       Train::Metrics& metrics) const
    {
        return metrics.add(Train::Metric{"perplexity", perplexity(step.loss), false});
    }

    Result<std::unique_ptr<Train::Objective>> make_next_token_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context) noexcept
    {
        SA_TRY_ASSIGN(std::string name, read_output_name(attributes, context));
        SA_FAIL_IF(given.input_sample.rank() != 1, StatusCode::InvalidDescription,
                   "{} reads [T] token ids per sample, but the model's input is {} per sample",
                   context, given.input_sample.to_string());
        SA_TRY_ASSIGN(const Train::OutputShape* const published,
                      find_logits(given, name, context));

        const Shape& logits = published->shape;
        SA_FAIL_IF(logits.rank() != K_LOGITS_RANK || logits[0] != given.batch
                   || logits[1] != given.input_sample[0],
                   StatusCode::InvalidDescription,
                   "{} reads [{}, {}, V] logits, but the model's output '{}' is {}", context,
                   given.batch, given.input_sample[0], published->name, logits.to_string());
        SA_FAIL_IF(given.classes > logits[2], StatusCode::InvalidDescription,
                   "{}: the data holds {} token ids but the model scores {}", context,
                   given.classes, logits[2]);

        SA_TRY_ASSIGN(std::unique_ptr<NextTokenObjective> made,
                      NextTokenObjective::make(logits[0], logits[1], logits[2],
                                               std::move(name)));
        std::unique_ptr<Train::Objective> base = std::move(made);
        return std::move(base);
    }

} // namespace SushiAI::Language::Training
