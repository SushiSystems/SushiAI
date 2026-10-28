/**************************************************************************/
/* language_head.cpp                                                      */
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
 * @file language_head.cpp
 * @brief Implements LanguageHead's trace, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/blocks/language_head.hpp>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/embedding.hpp>
#include "block_attributes.hpp"
#include "sequence.hpp"

namespace SushiAI
{
    namespace Language::Blocks
    {
        namespace
        {
            /** @brief Lists every key a "language_head" row accepts. */
            constexpr std::string_view K_LANGUAGE_HEAD_KEYS[] = {
                Detail::K_VOCABULARY_KEY, Detail::K_LAYER_NORM_EPSILON_KEY, Detail::K_TIE_KEY};

            /**
             * @brief Returns the embedding of the row "tie" names, or null for a row without
             *        the key.
             * @return InvalidDescription when the row is not above or is no "embedding" row.
             */
            [[nodiscard]] Result<NN::Embedding*> read_tied_embedding(
                const Config::Value& attributes, std::string_view context,
                const NN::BuiltRows& above)
            {
                if (!attributes.has(Detail::K_TIE_KEY))
                    return static_cast<NN::Embedding*>(nullptr);

                SA_TRY_ASSIGN(const std::size_t row,
                              Detail::read_row_above(attributes, Detail::K_TIE_KEY, context,
                                                     above.size()));
                NN::Embedding* const embedding = NN::find_embedding(above.node(row));
                SA_FAIL_IF(embedding == nullptr, StatusCode::InvalidDescription,
                           "{} '{}' names row {}, which is no 'embedding' row; a head ties to "
                           "the table of one",
                           context, Detail::K_TIE_KEY, row);

                return embedding;
            }

            /**
             * @brief Checks that the table @p head is tied to can score rows of @p dtype.
             * @pre head.tied_weight is not null.
             * @return InvalidDescription for a table no layer has declared, or one that is
             *         not [vocabulary, dim] of @p dtype.
             */
            [[nodiscard]] Status check_tied_table(const LanguageHead& head, DType dtype)
            {
                const NN::Parameter& table = *head.tied_weight;
                SA_FAIL_IF(!table.declared(), StatusCode::InvalidDescription,
                           "LanguageHead({}, {}) is tied to a table no layer has declared; "
                           "trace the embedding before the head",
                           head.dim, head.vocabulary);

                const Shape expected{head.vocabulary, head.dim};
                SA_FAIL_IF(table.shape() != expected || table.dtype() != dtype,
                           StatusCode::InvalidDescription,
                           "LanguageHead({}, {}) scores {} rows with a {} table of {}; it is "
                           "tied to a {} table of {}",
                           head.dim, head.vocabulary, dtype_name(dtype), dtype_name(dtype),
                           expected.to_string(), dtype_name(table.dtype()),
                           table.shape().to_string());

                return Status();
            }

            /** @brief Traces the head, or returns why @p x or @p head does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_head(LanguageHead& head,
                                                            const NN::Trace& trace,
                                                            Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FAIL_IF(head.dim <= 0 || head.vocabulary <= 0, StatusCode::InvalidDescription,
                           "LanguageHead needs positive widths; got dim={} vocabulary={}",
                           head.dim, head.vocabulary);
                SA_TRY_ASSIGN(const Shape sequence,
                              Detail::checked_sequence(builder, x, head.dim, "LanguageHead"));

                // Checked before the first node, so a refused table leaves the graph as it was.
                if (head.tied_weight != nullptr)
                    SA_TRY(check_tied_table(head, builder.graph().value(x).dtype));

                const Graph::ValueID normalised = NN::trace_layer(head.norm, trace, x);
                SA_TRY(trace.status());

                return Detail::over_rows(builder, normalised, sequence, head.vocabulary,
                                         [&](Graph::ValueID rows) -> Graph::Traced
                {
                    if (head.tied_weight == nullptr)
                    {
                        const Graph::ValueID logits = NN::trace_layer(head.linear, trace, rows);
                        SA_TRY(trace.status());

                        return logits;
                    }

                    // The table is [vocabulary, dim], which is a Linear weight's layout.
                    SA_TRY_ASSIGN(
                        const Graph::ValueID table,
                        head.tied_weight->declare(
                            builder, head.tied_weight->shape(), head.tied_weight->role(),
                            head.tied_weight->init(), head.tied_weight->fan_in(),
                            head.tied_weight->fan_out(), "weight", head.tied_weight->dtype()));
                    return builder.matmul(rows, table, false, true);
                });
            }
        } // namespace

        void LanguageHead::tie_to(NN::Parameter& table) noexcept
        {
            tied_weight = &table;
        }

        LanguageHead::LanguageHead(int64_t dim_in, int64_t vocabulary_in,
                                   double epsilon) noexcept
            : dim(dim_in), vocabulary(vocabulary_in)
        {
            norm.dim = dim;
            norm.epsilon = epsilon;
            linear.in_features = dim;
            linear.out_features = vocabulary;
            linear.use_bias = false;
        }

        Graph::ValueID LanguageHead::forward(const NN::Trace& trace, Graph::ValueID x) noexcept
        {
            return trace.value_or_fail(trace_head(*this, trace, x), x);
        }

        Graph::ValueID LanguageHead::forward_incremental(const NN::Trace& trace,
                                                         Graph::ValueID x, const Shape&) noexcept
        {
            return forward(trace, x);
        }

        Result<Shape> LanguageHead::output_shape(const Shape& input) const noexcept
        {
            SA_FATAL_IF(input.rank() != Detail::K_SAMPLE_RANK,
                        "LanguageHead::output_shape was given {}; a per-sample sequence is "
                        "[T, C]",
                        input.to_string());

            return Shape{input[0], vocabulary};
        }

        Result<std::unique_ptr<NN::Node>> make_language_head_node(NN::ShapeSpan inputs,
                                                                  const Config::Value& attributes,
                                                                  std::string_view context,
                                                                  const NN::BuiltRows& above) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_LANGUAGE_HEAD_KEYS),
                                               context));
            SA_TRY_ASSIGN(const Shape input, Detail::read_sequence(inputs, context));
            SA_TRY_ASSIGN(NN::Embedding* const tied,
                          read_tied_embedding(attributes, context, above));
            SA_TRY_ASSIGN(const double epsilon,
                          Detail::read_layer_norm_epsilon(attributes, context));
            if (tied == nullptr)
            {
                SA_TRY_ASSIGN(const int64_t vocabulary,
                              Detail::read_positive_int(attributes, Detail::K_VOCABULARY_KEY,
                                                        context));
                return Detail::wrap_block(LanguageHead(input[1], vocabulary, epsilon), input);
            }

            SA_TRY_ASSIGN(const int64_t vocabulary,
                          Detail::read_positive_int_or(attributes, Detail::K_VOCABULARY_KEY,
                                                       context, tied->num_embeddings));
            SA_FAIL_IF(tied->num_embeddings != vocabulary || tied->embedding_dim != input[1],
                       StatusCode::InvalidDescription,
                       "{} scores {} tokens from a width of {}, but the row '{}' names holds a "
                       "[{}, {}] table",
                       context, vocabulary, input[1], Detail::K_TIE_KEY, tied->num_embeddings,
                       tied->embedding_dim);

            // The table lives in its node, which the graph owns beside this one.
            LanguageHead head(input[1], vocabulary, epsilon);
            head.tie_to(tied->weight);
            return Detail::wrap_block(std::move(head), input);
        }

    } // namespace Language::Blocks
} // namespace SushiAI
