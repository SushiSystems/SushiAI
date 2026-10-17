/**************************************************************************/
/* decoder_model.hpp                                                      */
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
 * @file decoder_model.hpp
 * @brief Defines the small decoder model files the language tests build.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/language/registration/registration.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/model_text/model_text.hpp>

namespace SushiAI
{
    namespace Test
    {
        /** @brief Holds V, the tokens of the decoder. */
        inline constexpr std::int64_t K_DECODER_VOCABULARY = 32;

        /** @brief Holds T, the positions of the decoder's window. */
        inline constexpr std::int64_t K_DECODER_WINDOW = 8;

        /** @brief Holds D, the width of the decoder. */
        inline constexpr std::int64_t K_DECODER_DIM = 16;

        /** @brief Holds the blocks of the decoder. */
        inline constexpr std::int64_t K_DECODER_BLOCKS = 2;

        /** @brief Holds the model file: two blocks of two heads and a head tied to row 0. */
        inline constexpr std::string_view K_DECODER_MODEL = R"({
            "format_version": 2,
            "input": {"shape": [8], "dtype": "int32"},
            "layers": [
                {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
                {"type": "position_embedding"},
                {"type": "transformer_block", "heads": 2},
                {"type": "transformer_block", "heads": 2},
                {"type": "language_head", "tie": 0}
            ],
            "task": {"type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8}
        })";

        /** @brief Holds the positions of the window no generation of these tests fills. */
        inline constexpr std::int64_t K_WIDE_DECODER_WINDOW = 64;

        /** @brief Holds the decoder with rotary positions: two blocks and no position table. */
        inline constexpr std::string_view K_ROTARY_DECODER_MODEL = R"({
            "format_version": 2,
            "input": {"shape": [8], "dtype": "int32"},
            "layers": [
                {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
                {"type": "transformer_block", "heads": 2, "positions": "rotary"},
                {"type": "transformer_block", "heads": 2, "positions": "rotary"},
                {"type": "language_head", "tie": 0}
            ],
            "task": {"type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8}
        })";

        /** @brief Holds K_ROTARY_DECODER_MODEL over a window of K_WIDE_DECODER_WINDOW. */
        inline constexpr std::string_view K_WIDE_ROTARY_DECODER_MODEL = R"({
            "format_version": 2,
            "input": {"shape": [64], "dtype": "int32"},
            "layers": [
                {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
                {"type": "transformer_block", "heads": 2, "positions": "rotary"},
                {"type": "transformer_block", "heads": 2, "positions": "rotary"},
                {"type": "language_head", "tie": 0}
            ],
            "task": {"type": "next_token_generation", "vocabulary": "vocab.savb", "window": 64}
        })";

        /** @brief Holds the rotary decoder of one block, whose keys read no other position. */
        inline constexpr std::string_view K_SHALLOW_ROTARY_DECODER_MODEL = R"({
            "format_version": 2,
            "input": {"shape": [8], "dtype": "int32"},
            "layers": [
                {"type": "embedding", "num_embeddings": 32, "embedding_dim": 16},
                {"type": "transformer_block", "heads": 2, "positions": "rotary"},
                {"type": "language_head", "tie": 0}
            ],
            "task": {"type": "next_token_generation", "vocabulary": "vocab.savb", "window": 8}
        })";

        /** @brief Returns the parsed K_DECODER_MODEL. */
        [[nodiscard]] inline Config::ModelSpec decoder_spec()
        {
            return model_from(K_DECODER_MODEL);
        }

        /**
         * @brief Builds the decoder @p spec describes from the rows of nn and of language.
         * @return The graph, or the failure of a register function or of build_model.
         */
        [[nodiscard]] inline Result<NN::ModelGraph> build_decoder(const Config::ModelSpec& spec)
        {
            NN::NodeRegistry registry;
            SA_TRY(NN::register_layers(registry));
            SA_TRY(Language::register_nodes(registry));

            return NN::build_model(registry, spec);
        }

    } // namespace Test
} // namespace SushiAI
