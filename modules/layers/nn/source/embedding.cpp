/**************************************************************************/
/* embedding.cpp                                                          */
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
 * @file embedding.cpp
 * @brief Implements Embedding's trace form, its shape rule and its node factory.
 * @author Mustafa Garip
 */

#include <SushiAI/nn/embedding.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/nn/node_attributes.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

namespace SushiAI
{
    namespace NN
    {
        namespace
        {
            /** @brief Names the key that holds a row's table row count. */
            constexpr std::string_view K_NUM_EMBEDDINGS_KEY = "num_embeddings";

            /** @brief Names the key that holds a row's table row length. */
            constexpr std::string_view K_EMBEDDING_DIM_KEY = "embedding_dim";

            /** @brief Lists every key an "embedding" row accepts. */
            constexpr std::string_view K_EMBEDDING_KEYS[] = {K_NUM_EMBEDDINGS_KEY,
                                                             K_EMBEDDING_DIM_KEY};

            /** @brief Holds the highest rank of a per-sample index: a batch axis and D follow. */
            constexpr int32_t K_MAX_SAMPLE_RANK = K_MAX_RANK - 2;

            /** @brief Returns InvalidDescription unless both extents and the type fit. */
            [[nodiscard]] Status check_description(const Embedding& layer)
            {
                SA_FAIL_IF(layer.num_embeddings <= 0 || layer.embedding_dim <= 0,
                           StatusCode::InvalidDescription,
                           "Embedding needs positive extents; got num_embeddings={} "
                           "embedding_dim={}",
                           layer.num_embeddings, layer.embedding_dim);
                SA_FAIL_IF(!is_real_float(layer.dtype), StatusCode::InvalidDescription,
                           "Embedding({}, {}) needs a half, float32 or float64 table; got {}",
                           layer.num_embeddings, layer.embedding_dim, dtype_name(layer.dtype));

                return Status();
            }

            /** @brief Traces the gather, or returns why @p indices or @p layer does not fit. */
            [[nodiscard]] Result<Graph::ValueID> trace_gather(Embedding& layer, const Trace& trace,
                                                              Graph::ValueID indices)
            {
                const Graph::GraphBuilder& builder = trace.builder();
                SA_FATAL_IF(indices >= builder.graph().value_count(),
                            "Embedding({}, {}) was traced over value {}; the graph holds {} "
                            "values",
                            layer.num_embeddings, layer.embedding_dim, indices,
                            builder.graph().value_count());
                SA_TRY(check_description(layer));

                // The description is copied; declaring the table grows the value table.
                const Shape index_shape = builder.graph().value(indices).shape;
                const DType index_dtype = builder.graph().value(indices).dtype;
                SA_FAIL_IF(index_dtype != DType::Int32, StatusCode::InvalidDescription,
                           "Embedding({}, {}) needs an int32 input of indices; got {}",
                           layer.num_embeddings, layer.embedding_dim, dtype_name(index_dtype));
                SA_FAIL_IF(index_shape.rank() < 1 || index_shape.rank() >= K_MAX_RANK,
                           StatusCode::InvalidDescription,
                           "Embedding({}, {}) needs an input of rank 1 to {}; got {}",
                           layer.num_embeddings, layer.embedding_dim, K_MAX_RANK - 1,
                           index_shape.to_string());

                SA_TRY_ASSIGN(const Graph::ValueID table, Detail::bridge_throwing([&]()
                {
                    return layer.weight.declare(
                        builder, Shape{layer.num_embeddings, layer.embedding_dim},
                        ParameterRole::Embedding, layer.weight_init, layer.embedding_dim,
                        layer.embedding_dim, "weight", layer.dtype);
                }));

                const Graph::Traced gathered = builder.gather_rows(table, indices);
                SA_TRY(SA_ANNOTATE(gathered.status(), "Embedding({}, {})", layer.num_embeddings,
                                   layer.embedding_dim));

                return gathered.value();
            }
        } // namespace

        Graph::ValueID Embedding::forward(const Trace& trace, Graph::ValueID indices)
        {
            return trace.value_or_fail(trace_gather(*this, trace, indices), indices);
        }

        Graph::ValueID Embedding::forward_incremental(const Trace& trace, Graph::ValueID indices,
                                                      const Shape&)
        {
            return forward(trace, indices);
        }

        Shape Embedding::output_shape(const Shape& input) const
        {
            SA_FATAL_IF(input.rank() > K_MAX_SAMPLE_RANK,
                        "Embedding::output_shape was given {}; a per-sample index has a rank of "
                        "at most {}",
                        input.to_string(), K_MAX_SAMPLE_RANK);

            std::array<int64_t, K_MAX_RANK> dims{};
            const std::size_t rank = static_cast<std::size_t>(input.rank());
            for (std::size_t axis = 0; axis < rank; ++axis)
                dims[axis] = input[static_cast<int32_t>(axis)];

            dims[rank] = embedding_dim;
            return Shape(SushiRuntime::span<const int64_t>(dims.data(), rank + 1));
        }

        Embedding* find_embedding(Node& node) noexcept
        {
            // make_embedding_node wraps the layer through wrap_block, which is this type.
            auto* const wrapped = dynamic_cast<ModuleNode<Embedding>*>(&node);
            return wrapped == nullptr ? nullptr : &wrapped->layer();
        }

        Result<std::unique_ptr<Node>> make_embedding_node(ShapeSpan inputs,
                                                          const Config::Value& attributes,
                                                          std::string_view context)
        {
            SA_TRY(Detail::reject_unknown_keys(attributes, Detail::KeySpan(K_EMBEDDING_KEYS),
                                               context));
            SA_FAIL_IF(inputs.size() != 1, StatusCode::InvalidDescription,
                       "{} reads one input, but it was given {}", context, inputs.size());

            const Shape input = inputs[0];
            SA_FAIL_IF(input.rank() < 1 || input.rank() > K_MAX_SAMPLE_RANK,
                       StatusCode::InvalidDescription,
                       "{} was given a per-sample input of {}, but it reads indices of rank 1 "
                       "to {}",
                       context, input.to_string(), K_MAX_SAMPLE_RANK);

            Embedding layer{};
            SA_TRY_ASSIGN(layer.num_embeddings,
                          Detail::read_positive_int(attributes, K_NUM_EMBEDDINGS_KEY, context));
            SA_TRY_ASSIGN(layer.embedding_dim,
                          Detail::read_positive_int(attributes, K_EMBEDDING_DIM_KEY, context));

            return Detail::wrap_block(std::move(layer), input);
        }

    } // namespace NN
} // namespace SushiAI
