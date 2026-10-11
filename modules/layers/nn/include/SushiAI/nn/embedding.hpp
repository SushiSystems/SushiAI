/**************************************************************************/
/* embedding.hpp                                                          */
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
 * @file embedding.hpp
 * @brief Declares Embedding, the layer that reads rows of a learned table by an integer index.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <SushiAI/config/value.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/trace.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Holds the standard deviation an embedding table is drawn with. */
        inline constexpr double K_EMBEDDING_DEVIATION = 0.02;

        /** @brief Maps each INT32 index of its input to a row of a learned [V, D] table. */
        struct Embedding
        {
            /** @brief Holds V, the number of rows of the table; an index is in [0, V). */
            int64_t num_embeddings = 0;

            /** @brief Holds D, the length of one row, which is the last extent of the result. */
            int64_t embedding_dim = 0;

            /** @brief Names the element type of the table and of the result; a real float. */
            DType dtype = DType::Float32;

            /** @brief Holds the [num_embeddings, embedding_dim] table, of role Embedding. */
            Parameter weight{};

            /** @brief Holds the fill of the table: a normal of K_EMBEDDING_DEVIATION. */
            ParameterInit weight_init = ParameterInit::normal(K_EMBEDDING_DEVIATION);

            /** @brief Lists this layer's parameter for the state walk. */
            static constexpr auto fields()
            {
                return std::tuple{
                    field("weight", &Embedding::weight),
                };
            }

            /**
             * @brief Traces K_GATHER_ROWS over the table, declaring it on the first call.
             * @param indices An INT32 value of rank one or more, batch axis included.
             * @return The [..., embedding_dim] result's id, or @p indices after trace.fail()
             *         when @p indices or this layer's description does not fit.
             */
            Graph::ValueID forward(const Trace& trace, Graph::ValueID indices);

            /**
             * @brief Traces forward() over the newest indices, as a row reads no other index.
             * @param sample Not read.
             */
            Graph::ValueID forward_incremental(const Trace& trace, Graph::ValueID indices,
                                               const Shape& sample);

            /**
             * @brief Returns @p input with embedding_dim appended as its last extent.
             * @param input The per-sample shape of the indices.
             * @pre @p input has a rank of at most K_MAX_RANK - 2; a higher rank is fatal.
             */
            Shape output_shape(const Shape& input) const;
        };

        static_assert(TraceModule<Embedding>, "Embedding must satisfy the TraceModule contract");
        static_assert(IncrementalModule<Embedding>, "Embedding must have an incremental form");

        /**
         * @brief Returns the Embedding that @p node holds, when make_embedding_node built it.
         * @return The layer, which lives at its address as long as @p node, or null for a
         *         node of any other kind.
         */
        [[nodiscard]] Embedding* find_embedding(Node& node) noexcept;

        /**
         * @brief Builds an Embedding node from the keys "num_embeddings" and "embedding_dim".
         * @param inputs One per-sample shape of rank 1 to K_MAX_RANK - 2, the shape of the
         *        indices.
         * @param context What to call the row in a diagnostic, e.g. "layer 0 'embedding'".
         * @return The node, or InvalidDescription when @p inputs or a key does not fit.
         */
        [[nodiscard]] Result<std::unique_ptr<Node>> make_embedding_node(
            ShapeSpan inputs, const Config::Value& attributes, std::string_view context);

    } // namespace NN
} // namespace SushiAI
