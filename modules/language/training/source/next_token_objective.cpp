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
 * @file next_token_objective.cpp
 * @brief Implements the next-token objective and the factory a training file reaches it by.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/next_token_objective.hpp>

#include <cstdint>
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
#include <SushiAI/language/training/next_token_inputs.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>
#include <SushiAI/train/batch_role.hpp>

namespace SushiAI::Language::Training
{
    namespace
    {
        /** @brief Names the rank of the [N, T, V] logits the objective reads. */
        constexpr std::int32_t K_LOGITS_RANK = 3;

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
        std::int64_t batch, std::int64_t length, std::int64_t vocabulary, std::string output,
        std::int64_t input_classes) noexcept
    {
        SA_FAIL_IF(batch < 1 || length < 1 || vocabulary < 1, StatusCode::InvalidDescription,
                   "A next-token objective needs a batch, a length and a vocabulary of at "
                   "least 1; got {}, {} and {}",
                   batch, length, vocabulary);

        SA_FAIL_IF(input_classes < 0, StatusCode::InvalidDescription,
                   "A next-token objective bounds its input ids by the rows of the embedding, "
                   "0 or more; got {}", input_classes);

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<NextTokenObjective>(
                new NextTokenObjective(batch, length, vocabulary, std::move(output),
                                       input_classes));
        });
    }

    NextTokenObjective::NextTokenObjective(std::int64_t batch, std::int64_t length,
                                           std::int64_t vocabulary,
                                           std::string output,
                                           std::int64_t input_classes) noexcept
        : batch_(batch), length_(length), vocabulary_(vocabulary), output_(std::move(output)),
          input_classes_(input_classes)
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

    Result<Train::LossTrace> NextTokenObjective::loss(
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

        // The label of each position is the column its log-probability sits at.
        const Graph::Traced picked =
            builder.take_along(log_probabilities, leaves.at(K_NEXT_TOKEN_TARGETS_LEAF));
        const Graph::Traced picked_column = builder.reshape(picked, Shape{rows, 1});
        SA_TRY_ASSIGN(const Graph::ValueID loss,
                      builder.scale(builder.sum_rows(picked_column),
                                    -1.0 / static_cast<double>(rows)));

        return Train::LossTrace(loss);
    }

    Status NextTokenObjective::feed(const Data::Batch& batch,
                                    const Train::ObjectiveLeaves& leaves,
                                    Graph::TensorPool& pool) const
    {
        SA_TRY_ASSIGN(const Data::BatchTarget* const labels,
                      Train::require_batch_role(batch, Data::K_ROLE_LABELS));
        SA_TRY(check_next_token_inputs(batch.inputs,
                                       input_classes_ != 0 ? input_classes_ : vocabulary_));
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
                                               std::move(name), given.input_classes));
        std::unique_ptr<Train::Objective> base = std::move(made);
        return std::move(base);
    }

} // namespace SushiAI::Language::Training
