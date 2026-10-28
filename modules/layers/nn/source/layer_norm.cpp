/**************************************************************************/
/* layer_norm.cpp                                                         */
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
 * @file layer_norm.cpp
 * @brief Implements LayerNorm's trace form, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/layer_norm.hpp>

#include <utility>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/nn/node_attributes.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Lists every key a "layernorm" row accepts. */
            constexpr std::string_view K_LAYER_NORM_KEYS[] = {Detail::K_LAYER_NORM_EPSILON_KEY};

            /** @brief Holds the highest rank of a per-sample input: a batch axis precedes it. */
            constexpr int32_t K_MAX_SAMPLE_RANK = K_MAX_RANK - 1;

            /** @brief Returns InvalidDescription unless the extent and the epsilon fit. */
            [[nodiscard]] Status check_description(const LayerNorm& layer)
            {
                SA_FAIL_IF(layer.dim <= 0, StatusCode::InvalidDescription,
                           "LayerNorm needs a positive dim; got {}", layer.dim);
                SA_FAIL_IF(!(layer.epsilon > 0.0), StatusCode::InvalidDescription,
                           "LayerNorm({}) needs a positive epsilon; got {}", layer.dim,
                           layer.epsilon);

                return Status();
            }

            /** @brief Traces the node, or returns why @p x or @p layer does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_normalised(LayerNorm& layer,
                                                                  const Trace& trace,
                                                                  Graph::ValueID x)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FATAL_IF(x >= builder.graph().value_count(),
                            "LayerNorm({}) was traced over value {}; the graph holds {} values",
                            layer.dim, x, builder.graph().value_count());
                SA_TRY(check_description(layer));

                // The description is copied; declaring a parameter grows the value table.
                const Shape shape = builder.graph().value(x).shape;
                const DType dtype = builder.graph().value(x).dtype;
                SA_FAIL_IF(shape.rank() < 1 || shape[shape.rank() - 1] != layer.dim,
                           StatusCode::InvalidDescription,
                           "LayerNorm({}) needs a [..., {}] input; got {}", layer.dim, layer.dim,
                           shape.to_string());
                SA_FAIL_IF(!is_real_float(dtype), StatusCode::InvalidDescription,
                           "LayerNorm({}) needs a half, float32 or float64 input; got {}",
                           layer.dim, dtype_name(dtype));

                const Shape per_feature{layer.dim};
                SA_TRY_ASSIGN(const Graph::ValueID gamma,
                              layer.weight.declare(builder, per_feature, ParameterRole::Scale,
                                                   layer.dim, layer.dim, "weight", dtype));
                SA_TRY_ASSIGN(const Graph::ValueID beta,
                              layer.bias.declare(builder, per_feature, ParameterRole::Bias,
                                                 layer.dim, layer.dim, "bias", dtype));

                const Result<Graph::LayerNormResult> normalised =
                    builder.layer_norm(x, gamma, beta, layer.epsilon);
                SA_TRY(SA_ANNOTATE(normalised.status(), "LayerNorm({})", layer.dim));

                SA_TRY_ASSIGN(const Graph::LayerNormResult result, normalised);
                return result.output;
            }
        } // namespace

        Graph::ValueID LayerNorm::forward(const Trace& trace, Graph::ValueID x) noexcept
        {
            return trace.value_or_fail(trace_normalised(*this, trace, x), x);
        }

        Graph::ValueID LayerNorm::forward_incremental(const Trace& trace, Graph::ValueID x,
                                                      const Shape&) noexcept
        {
            return forward(trace, x);
        }

        Result<Shape> LayerNorm::output_shape(const Shape& input) const noexcept
        {
            return input;
        }

        Result<std::unique_ptr<Node>> make_layer_norm_node(ShapeSpan inputs,
                                                           const Config::Value& attributes,
                                                           std::string_view context) noexcept
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_LAYER_NORM_KEYS),
                                               context));
            SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                       "{} reads one input, but it was given {}", context, inputs.size());

            const Shape input = inputs[0];
            SA_FAIL_IF(input.rank() < 1 || input.rank() > K_MAX_SAMPLE_RANK,
                       StatusCode::InvalidDescription,
                       "{} was given a per-sample input of {}, but it reads an input of rank 1 "
                       "to {}",
                       context, input.to_string(), K_MAX_SAMPLE_RANK);

            LayerNorm layer{};
            layer.dim = input[input.rank() - 1];
            SA_TRY_ASSIGN(layer.epsilon,
                          Detail::read_layer_norm_epsilon(attributes, context));

            return Detail::wrap_block(std::move(layer), input);
        }

    } // namespace NN
} // namespace SushiAI
