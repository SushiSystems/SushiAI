/**************************************************************************/
/* registration.cpp                                                       */
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
 * @file registration.cpp
 * @brief Implements the functions that add the rows of language to the registries.
 * @author Mustafa Garip
 */

#include <SushiAI/language/registration.hpp>

#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/datasets/tokens_section.hpp>
#include <SushiAI/language/training/expert_balance_objective.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/train/objective_registry.hpp>

namespace SushiAI::Language
{
    Status register_nodes(NN::NodeRegistry& registry)
    {
        SA_TRY(registry.add(NN::NodeEntry{"position_embedding",
                                          &Blocks::make_position_embedding_node}));
        SA_TRY(registry.add(NN::NodeEntry{"causal_self_attention",
                                          &Blocks::make_causal_self_attention_node}));
        SA_TRY(registry.add(NN::NodeEntry{"feed_forward", &Blocks::make_feed_forward_node}));
        SA_TRY(registry.add(NN::NodeEntry{"transformer_block",
                                          &Blocks::make_transformer_block_node}));
        SA_TRY(registry.add(NN::NodeEntry{"moe_feed_forward",
                                          &Blocks::make_moe_feed_forward_node}));

        // The head may name the embedding row above it, so its factory reads the rows.
        return registry.add(NN::LinkedNodeEntry{"language_head",
                                                &Blocks::make_language_head_node});
    }

    Status register_datasets(Data::DatasetRegistry& registry)
    {
        SA_TRY(registry.add(Data::DatasetEntry{Datasets::K_TOKENS_DATA_TYPE,
                                               &Datasets::make_tokens_dataset}));

        return registry.add(Data::DatasetEntry{Datasets::K_PROMPT_DATA_TYPE,
                                               &Datasets::make_prompt_dataset});
    }

    Status register_objectives(Train::ObjectiveRegistry& registry)
    {
        SA_TRY(registry.add(Train::ObjectiveEntry{Training::K_NEXT_TOKEN_OBJECTIVE_TYPE,
                                                  &Training::make_next_token_objective}));

        return registry.add(Train::ObjectiveEntry{Training::K_EXPERT_BALANCE_OBJECTIVE_TYPE,
                                                  &Training::make_expert_balance_objective});
    }

} // namespace SushiAI::Language
