/**************************************************************************/
/* language.h                                                             */
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
 * @file language.h
 * @brief Includes SushiAI and every public header of the language domain.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/SushiAI.h>

#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>

#include <SushiAI/language/datasets/collator.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>

#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>

#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>

#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/datasets/tokens_section.hpp>
#include <SushiAI/language/generation/generation_task.hpp>
#include <SushiAI/language/generation/greedy_sampler.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/sampler_section.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/registration.hpp>
#include <SushiAI/language/text/stream_encoder.hpp>

#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/position_feed.hpp>
#include <SushiAI/language/generation/ring_position_feed.hpp>
#include <SushiAI/language/generation/table_position_feed.hpp>

#include <SushiAI/language/blocks/expert_projection.hpp>
#include <SushiAI/language/blocks/expert_trace.hpp>
#include <SushiAI/language/blocks/expert_transformer_block.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/language/training/expert_balance_objective.hpp>
