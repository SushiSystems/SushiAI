/**************************************************************************/
/* test_noexcept_surface.cpp                                              */
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
 * @file test_noexcept_surface.cpp
 * @brief Tests that every public function of the language modules is noexcept.
 * @author Mustafa Garip
 */

#include <utility>

#include <gtest/gtest.h>

#include <SushiAI/language/blocks/causal_self_attention.hpp>
#include <SushiAI/language/blocks/expert_projection.hpp>
#include <SushiAI/language/blocks/expert_transformer_block.hpp>
#include <SushiAI/language/blocks/feed_forward.hpp>
#include <SushiAI/language/blocks/language_head.hpp>
#include <SushiAI/language/blocks/moe_feed_forward.hpp>
#include <SushiAI/language/blocks/position_embedding.hpp>
#include <SushiAI/language/blocks/rotary_embedding.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/language/blocks/transformer_block.hpp>
#include <SushiAI/language/datasets/next_token_collator.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/datasets/token_dataset.hpp>
#include <SushiAI/language/datasets/token_file.hpp>
#include <SushiAI/language/datasets/tokens_section.hpp>
#include <SushiAI/language/datasets/window_layout.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generation_task.hpp>
#include <SushiAI/language/generation/nucleus_sampler.hpp>
#include <SushiAI/language/generation/sampler_section.hpp>
#include <SushiAI/language/generation/temperature_sampler.hpp>
#include <SushiAI/language/generation/top_k_sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/language/registration/registration.hpp>
#include <SushiAI/language/text/bpe_trainer.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/stream_encoder.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>
#include <SushiAI/language/training/expert_balance_objective.hpp>
#include <SushiAI/language/training/next_token_objective.hpp>
#include <SushiAI/language/training/next_token_targets.hpp>
#include <SushiAI/language/training/perplexity.hpp>

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(CausalSelfAttention(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<AttentionMask>(), std::declval<AttentionPositions>())),
        "CausalSelfAttention is noexcept");
    static_assert(noexcept(std::declval<CausalSelfAttention&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<CausalSelfAttention&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<CausalSelfAttention&>().forward_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const CausalSelfAttention&>().output_shape(std::declval<const Shape&>(
                ))),
        "output_shape is noexcept");
    static_assert(
        noexcept(check_attention_widths(std::declval<int64_t>(), std::declval<int64_t>())),
        "check_attention_widths is noexcept");
    static_assert(
        noexcept(check_attention_positions(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<AttentionPositions>())),
        "check_attention_positions is noexcept");
    static_assert(
        noexcept(make_causal_self_attention_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_causal_self_attention_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(noexcept(std::declval<ExpertProjection&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<ExpertProjection&>().mix(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>(), std::declval<const Graph::Traced&>())),
        "mix is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(ExpertTransformerBlock(std::declval<int64_t>(), std::declval<MoEFeedForward>(),
                std::declval<AttentionMask>(), std::declval<double>(),
                std::declval<AttentionPositions>())),
        "ExpertTransformerBlock is noexcept");
    static_assert(noexcept(std::declval<ExpertTransformerBlock&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<ExpertTransformerBlock&>().trace_experts(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>())),
        "trace_experts is noexcept");
    static_assert(
        noexcept(std::declval<ExpertTransformerBlock&>().trace_experts_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "trace_experts_incremental is noexcept");
    static_assert(
        noexcept(std::declval<ExpertTransformerBlock&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<ExpertTransformerBlock&>().forward_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const ExpertTransformerBlock&>().output_shape(
                std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(FeedForward(std::declval<int64_t>(), std::declval<int64_t>())),
        "FeedForward is noexcept");
    static_assert(noexcept(std::declval<FeedForward&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<FeedForward&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<FeedForward&>().forward_incremental(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const FeedForward&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_feed_forward_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_feed_forward_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(LanguageHead(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<double>())),
        "LanguageHead is noexcept");
    static_assert(
        noexcept(std::declval<LanguageHead&>().tie_to(std::declval<NN::Parameter&>())),
        "tie_to is noexcept");
    static_assert(noexcept(std::declval<LanguageHead&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<LanguageHead&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<LanguageHead&>().forward_incremental(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const LanguageHead&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_language_head_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>(),
                std::declval<const NN::BuiltRows&>())),
        "make_language_head_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(MoEFeedForward(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "MoEFeedForward is noexcept");
    static_assert(noexcept(std::declval<MoEFeedForward&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<MoEFeedForward&>().trace_experts(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_experts is noexcept");
    static_assert(
        noexcept(std::declval<MoEFeedForward&>().trace_experts_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "trace_experts_incremental is noexcept");
    static_assert(
        noexcept(std::declval<MoEFeedForward&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<MoEFeedForward&>().forward_incremental(std::declval<const NN::Trace&>(
                ), std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const MoEFeedForward&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(check_expert_counts(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "check_expert_counts is noexcept");
    static_assert(
        noexcept(make_moe_feed_forward_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_moe_feed_forward_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(PositionEmbedding(std::declval<int64_t>(), std::declval<int64_t>())),
        "PositionEmbedding is noexcept");
    static_assert(noexcept(std::declval<PositionEmbedding&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<PositionEmbedding&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<PositionEmbedding&>().forward_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const PositionEmbedding&>().output_shape(std::declval<const Shape&>(
                ))),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_position_embedding_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_position_embedding_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(RotaryEmbedding(std::declval<int64_t>(), std::declval<int64_t>())),
        "RotaryEmbedding is noexcept");
    static_assert(noexcept(std::declval<RotaryEmbedding&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<RotaryEmbedding&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<RotaryEmbedding&>().forward_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const RotaryEmbedding&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(check_rotary_widths(std::declval<int64_t>(), std::declval<int64_t>())),
        "check_rotary_widths is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(rotary_angle(std::declval<std::int64_t>(), std::declval<std::int64_t>(),
                std::declval<std::int64_t>())),
        "rotary_angle is noexcept");
    static_assert(
        noexcept(write_rotary_row(std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "write_rotary_row is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Blocks
{
    static_assert(
        noexcept(TransformerBlock(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<AttentionMask>(), std::declval<double>(),
                std::declval<AttentionPositions>())),
        "TransformerBlock is noexcept");
    static_assert(noexcept(std::declval<TransformerBlock&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<TransformerBlock&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<TransformerBlock&>().forward_incremental(
                std::declval<const NN::Trace&>(), std::declval<Graph::ValueID>(),
                std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const TransformerBlock&>().output_shape(std::declval<const Shape&>(
                ))),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_transformer_block_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_transformer_block_node is noexcept");
} // namespace SushiAI::Language::Blocks

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(std::declval<NextTokenCollator&>().make(std::declval<std::int64_t>(),
                std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(std::declval<PromptDataset&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const std::vector<Text::TokenID>&>(), std::declval<std::int64_t>())),
        "make is noexcept");
    static_assert(
        noexcept(make_prompt_dataset(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_prompt_dataset is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(std::declval<TokenDataset&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::filesystem::path>(),
                std::declval<std::unique_ptr<const Collator>>(),
                std::declval<TokenDatasetOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const TokenDataset&>().vocabulary_size()),
        "vocabulary_size is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(write_token_file(std::declval<const std::filesystem::path&>(),
                std::declval<SushiRuntime::span<const Text::TokenID>>(),
                std::declval<std::int32_t>())),
        "write_token_file is noexcept");
    static_assert(
        noexcept(read_token_file_header(std::declval<const std::filesystem::path&>())),
        "read_token_file_header is noexcept");
    static_assert(
        noexcept(read_token_range(std::declval<std::istream&>(),
                std::declval<const TokenFileHeader&>(), std::declval<std::uint64_t>(),
                std::declval<SushiRuntime::span<Text::TokenID>>())),
        "read_token_range is noexcept");
    static_assert(
        noexcept(read_token_range(std::declval<const std::filesystem::path&>(),
                std::declval<const TokenFileHeader&>(), std::declval<std::uint64_t>(),
                std::declval<SushiRuntime::span<Text::TokenID>>())),
        "read_token_range is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(make_tokens_dataset(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_tokens_dataset is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Datasets
{
    static_assert(
        noexcept(count_windows(std::declval<std::uint64_t>(), std::declval<const WindowLayout&>())),
        "count_windows is noexcept");
    static_assert(
        noexcept(first_token_of(std::declval<std::uint64_t>(),
                std::declval<const WindowLayout&>())),
        "first_token_of is noexcept");
} // namespace SushiAI::Language::Datasets

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(std::declval<CachedGenerator&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<Train::EvaluationStep&>(), std::declval<std::string_view>(),
                std::declval<const CachedStepLeaves&>())),
        "make is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(read_generation_task(std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "read_generation_task is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(std::declval<NucleusSampler&>().make(std::declval<double>(),
                std::declval<double>())),
        "make is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(make_sampler(std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_sampler is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(std::declval<TemperatureSampler&>().make(std::declval<double>())),
        "make is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(std::declval<TopKSampler&>().make(std::declval<std::int64_t>(),
                std::declval<double>())),
        "make is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language::Generation
{
    static_assert(
        noexcept(std::declval<WindowGenerator&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<Train::EvaluationStep&>(), std::declval<std::string_view>())),
        "make is noexcept");
} // namespace SushiAI::Language::Generation

namespace SushiAI::Language
{
    static_assert(
        noexcept(register_nodes(std::declval<NN::NodeRegistry&>())),
        "register_nodes is noexcept");
    static_assert(
        noexcept(register_datasets(std::declval<Data::DatasetRegistry&>())),
        "register_datasets is noexcept");
    static_assert(
        noexcept(register_objectives(std::declval<Train::ObjectiveRegistry&>())),
        "register_objectives is noexcept");
} // namespace SushiAI::Language

namespace SushiAI::Language::Text
{
    static_assert(
        noexcept(train_bpe(std::declval<std::string_view>(),
                std::declval<const BPETrainerOptions&>())),
        "train_bpe is noexcept");
} // namespace SushiAI::Language::Text

namespace SushiAI::Language::Text
{
    static_assert(
        noexcept(encode_text(std::declval<const Vocabulary&>(), std::declval<std::string_view>(),
                std::declval<std::vector<TokenID>&>())),
        "encode_text is noexcept");
    static_assert(
        noexcept(decode_tokens(std::declval<const Vocabulary&>(),
                std::declval<SushiRuntime::span<const TokenID>>(), std::declval<std::string&>())),
        "decode_tokens is noexcept");
} // namespace SushiAI::Language::Text

namespace SushiAI::Language::Text
{
    static_assert(
        noexcept(encode_stream(std::declval<const Vocabulary&>(), std::declval<std::istream&>(),
                std::declval<std::size_t>(), std::declval<std::vector<TokenID>&>())),
        "encode_stream is noexcept");
} // namespace SushiAI::Language::Text

namespace SushiAI::Language::Text
{
    static_assert(
        noexcept(std::declval<Vocabulary&>().make(std::declval<std::vector<Merge>>(),
                std::declval<std::vector<std::string>>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const Vocabulary&>().size()), "size is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().contains(std::declval<TokenID>())),
        "contains is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().first_merged()),
        "first_merged is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().kind_of(std::declval<TokenID>())),
        "kind_of is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().bytes_of(std::declval<TokenID>())),
        "bytes_of is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().find_merge(std::declval<TokenID>(),
                std::declval<TokenID>())),
        "find_merge is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().find_special(std::declval<std::string_view>())),
        "find_special is noexcept");
    static_assert(noexcept(std::declval<const Vocabulary&>().merges()), "merges is noexcept");
    static_assert(
        noexcept(std::declval<const Vocabulary&>().special_names()),
        "special_names is noexcept");
} // namespace SushiAI::Language::Text

namespace SushiAI::Language::Text
{
    static_assert(
        noexcept(write_vocabulary_file(std::declval<const std::filesystem::path&>(),
                std::declval<const Vocabulary&>())),
        "write_vocabulary_file is noexcept");
    static_assert(
        noexcept(parse_vocabulary_file(std::declval<std::string_view>())),
        "parse_vocabulary_file is noexcept");
    static_assert(
        noexcept(read_vocabulary_file(std::declval<const std::filesystem::path&>())),
        "read_vocabulary_file is noexcept");
} // namespace SushiAI::Language::Text

namespace SushiAI::Language::Training
{
    static_assert(
        noexcept(std::declval<ExpertBalanceObjective&>().make(
                std::declval<std::vector<std::string>>(),
                std::declval<std::vector<ExpertUsageOutput>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const ExpertBalanceObjective&>().balances()),
        "balances is noexcept");
    static_assert(
        noexcept(std::declval<const ExpertBalanceObjective&>().usages()),
        "usages is noexcept");
    static_assert(
        noexcept(make_expert_balance_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_expert_balance_objective is noexcept");
} // namespace SushiAI::Language::Training

namespace SushiAI::Language::Training
{
    static_assert(
        noexcept(std::declval<NextTokenObjective&>().make(std::declval<std::int64_t>(),
                std::declval<std::int64_t>(), std::declval<std::int64_t>(),
                std::declval<std::string>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const NextTokenObjective&>().batch()), "batch is noexcept");
    static_assert(
        noexcept(std::declval<const NextTokenObjective&>().length()),
        "length is noexcept");
    static_assert(
        noexcept(std::declval<const NextTokenObjective&>().vocabulary()),
        "vocabulary is noexcept");
    static_assert(
        noexcept(make_next_token_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_next_token_objective is noexcept");
} // namespace SushiAI::Language::Training

namespace SushiAI::Language::Training
{
    static_assert(
        noexcept(fill_next_token_targets(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::int64_t>(), std::declval<SushiBLAS::Tensor&>())),
        "fill_next_token_targets is noexcept");
} // namespace SushiAI::Language::Training

namespace SushiAI::Language::Training
{
    static_assert(noexcept(perplexity(std::declval<double>())), "perplexity is noexcept");
    static_assert(
        noexcept(next_token_metrics(std::declval<double>())),
        "next_token_metrics is noexcept");
} // namespace SushiAI::Language::Training

/** @brief Passes when the file compiled: the assertions above are the test. */
TEST(Small_NoexceptSurface, EveryPublicFunctionIsNoexcept)
{
    SUCCEED();
}
