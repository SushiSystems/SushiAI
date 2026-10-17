/**************************************************************************/
/* domains_language.cpp                                                   */
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
 * @file domains_language.cpp
 * @brief Adds the rows of language to the app registries.
 * @author Mustafa Garip
 * @see applications/programs/README.md, "The run table" and "The report table".
 */

#include "domains.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <SushiLog/text/text_format.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/language/blocks/incremental.hpp>
#include <SushiAI/language/blocks/rotary_positions.hpp>
#include <SushiAI/language/datasets/prompt_dataset.hpp>
#include <SushiAI/language/generation/cached_generator.hpp>
#include <SushiAI/language/generation/generation_task.hpp>
#include <SushiAI/language/generation/generator.hpp>
#include <SushiAI/language/generation/sampler.hpp>
#include <SushiAI/language/generation/window_generator.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/language/registration/registration.hpp>
#include <SushiAI/language/text/codec.hpp>
#include <SushiAI/language/text/token.hpp>
#include <SushiAI/language/text/vocabulary.hpp>
#include <SushiAI/language/text/vocabulary_file.hpp>

namespace SushiAI
{
    namespace Apps
    {
        namespace
        {
            /** @brief Shortens the namespace of the generation module. */
            namespace Generation = Language::Generation;

            /** @brief Shortens the namespace of the tokenizer. */
            namespace Text = Language::Text;

            /** @brief Views a run of token ids. */
            using TokenSpan = SushiRuntime::span<const Text::TokenID>;

            /** @brief Names the task section in a diagnostic. */
            constexpr std::string_view K_TASK_CONTEXT = "the task section";

            /**
             * @brief Reads the ids of the prompt from batch 0 of the data.
             * @return A view into @p batch, or InvalidDescription for data that holds no
             *         batch or no prompt length, which a "prompt" data block states.
             */
            [[nodiscard]] Result<TokenSpan> read_prompt(const Data::Dataset& dataset,
                                                        Data::Batch& batch)
            {
                SA_FAIL_IF(dataset.batch_count() == 0, StatusCode::InvalidDescription,
                           "the '{}' data holds no batch, so there is no prompt",
                           dataset.name());
                SA_TRY(dataset.read(0, batch));

                const Data::BatchTarget* const length =
                    Data::find_target(batch.targets, Language::Datasets::K_ROLE_PROMPT_LENGTH);
                SA_FAIL_IF(length == nullptr, StatusCode::InvalidDescription,
                           "the '{}' data holds no '{}' role; a generation reads a data block "
                           "of the type '{}'",
                           dataset.name(), Language::Datasets::K_ROLE_PROMPT_LENGTH,
                           Language::Datasets::K_PROMPT_DATA_TYPE);

                const std::int32_t count = length->values.data_as<std::int32_t>()[0];
                return TokenSpan(batch.inputs.data_as<Text::TokenID>(),
                                 static_cast<std::size_t>(count));
            }

            /** @brief Holds a generator and what it runs on; the generator is released first. */
            struct BuiltGenerator
            {
                /** @brief Holds the values of a step the row compiled; null for the run's own. */
                std::unique_ptr<Graph::TensorPool> pool;

                /** @brief Holds the step the row compiled; null for the run's own. */
                std::unique_ptr<Train::EvaluationStep> step;

                /** @brief Holds the generator; never null in a built row. */
                std::unique_ptr<Generation::Generator> generator;
            };

            /** @brief Builds the generator of one row over what the run prepared. */
            using GeneratorFactory = Result<BuiltGenerator> (*)(PreparedInference& run);

            /** @brief Holds one row of the generator table. */
            struct GeneratorRow
            {
                bool cache;            /**< @brief The "cache" of the task the row serves. */
                std::string_view name; /**< @brief What the report calls the generator. */
                GeneratorFactory make; /**< @brief What builds the generator. */
            };

            /** @brief Builds the generator that runs the run's whole window for each token. */
            [[nodiscard]] Result<BuiltGenerator> make_window_generator(PreparedInference& run)
            {
                BuiltGenerator built;
                SA_TRY_ASSIGN(built.generator,
                              Generation::WindowGenerator::make(run.engine, run.step,
                                                                run.outputs.front().first));
                return built;
            }

            /**
             * @brief Traces one position of the run's model, compiles it and builds the
             *        generator that carries the keys and the values between runs.
             * @return Unsupported for a run that holds no built model or state.
             */
            [[nodiscard]] Result<BuiltGenerator> make_cached_generator(PreparedInference& run)
            {
                SA_FAIL_IF(run.built_model == nullptr || run.state == nullptr,
                           StatusCode::Unsupported,
                           "'{}' asks for the key and value cache, but the run holds no built "
                           "model and state to trace one position from; state \"cache\": false "
                           "in its task section to run the whole window",
                           run.model_path.string());

                Train::GraphOptions options;
                options.dtype = run.model.input_dtype;
                SA_TRY_ASSIGN(const Train::TracedStep traced,
                              Train::build_incremental_graph(*run.built_model, Shape{1, 1},
                                                             options));
                SA_FAIL_IF(traced.outputs.all().size() != 1, StatusCode::InvalidDescription,
                           "'{}' publishes {} outputs in its one-position form, but a "
                           "generation reads the logits of one",
                           run.model_path.string(), traced.outputs.all().size());

                BuiltGenerator built;
                SA_TRY_ASSIGN(built.pool,
                              Detail::guard_value(Detail::SeamPhase::Allocate, [&run]()
                {
                    return std::make_unique<Graph::TensorPool>(run.engine);
                }));
                SA_TRY_ASSIGN(built.step, Train::EvaluationStep::make(run.engine, traced,
                                                                      *run.state, *built.pool));

                // The generator reads whichever of the position and the ring the step is fed.
                const Generation::CachedStepLeaves leaves{
                    Language::Blocks::K_POSITION_LEAF, Language::Blocks::K_ATTENTION_MASK_LEAF,
                    Language::Blocks::K_MASKED_ATTENTION_SCORE,
                    Generation::RingStepLeaves{Language::Blocks::K_CACHE_SLOT_LEAF,
                                               Language::Blocks::K_ROTARY_COSINE_LEAF,
                                               Language::Blocks::K_ROTARY_SINE_LEAF,
                                               &Language::Blocks::write_rotary_row}};
                SA_TRY_ASSIGN(built.generator,
                              Generation::CachedGenerator::make(
                                  run.engine, *built.step, traced.outputs.all().front().first,
                                  leaves));
                return built;
            }

            /** @brief Lists the generators a task section may ask for, by its "cache". */
            constexpr GeneratorRow K_GENERATORS[] = {
                {true, "one position per token, keys and values cached", &make_cached_generator},
                {false, "the whole window per token", &make_window_generator},
            };

            /** @brief Returns the row whose "cache" is @p cache; the table holds both. */
            [[nodiscard]] const GeneratorRow& generator_row(bool cache) noexcept
            {
                for (const GeneratorRow& row : K_GENERATORS)
                {
                    if (row.cache == cache)
                        return row;
                }

                SA_FATAL("the generator table holds no row for cache={}", cache);
            }

            /**
             * @brief Continues the prompt of the data and prints the prompt and what follows.
             * @return Success, or the first failure of the task section, the vocabulary
             *         file, the data or a run of the step.
             */
            [[nodiscard]] Status report_generation(PreparedInference& run)
            {
                SA_TRY_ASSIGN(const Generation::GenerationTask task,
                              Generation::read_generation_task(run.model.task, K_TASK_CONTEXT));
                SA_FAIL_IF(run.model.input_shape != Shape{task.window},
                           StatusCode::InvalidDescription,
                           "'{}' states a window of {} in its task section, but its input "
                           "shape is {}",
                           run.model_path.string(), task.window,
                           run.model.input_shape.to_string());
                SA_FAIL_IF(run.outputs.size() != 1, StatusCode::InvalidDescription,
                           "'{}' publishes {} outputs, but a generation reads the logits of one",
                           run.model_path.string(), run.outputs.size());

                // A relative path is resolved against the directory of the model file.
                const std::filesystem::path vocabulary_path =
                    run.model_path.parent_path() / task.vocabulary;
                SA_TRY_ASSIGN(const Text::Vocabulary vocabulary,
                              Text::read_vocabulary_file(vocabulary_path));

                const GeneratorRow& row = generator_row(task.cache);
                SA_TRY_ASSIGN(const BuiltGenerator built, row.make(run));
                Generation::Generator* const generator = built.generator.get();
                SA_FAIL_IF(generator->vocabulary() < vocabulary.size(),
                           StatusCode::InvalidDescription,
                           "'{}' scores {} tokens, but '{}' holds {}, which is more",
                           run.model_path.string(), generator->vocabulary(),
                           vocabulary_path.string(), vocabulary.size());

                Data::Batch batch;
                SA_TRY_ASSIGN(const TokenSpan prompt, read_prompt(run.dataset, batch));

                Generation::RandomGenerator random(task.seed);
                std::vector<Text::TokenID> generated;
                SA_TRY(generator->generate(prompt, task.tokens, *task.sampler, random,
                                           generated));

                std::string text;
                SA_TRY(Text::decode_tokens(vocabulary, prompt, text));
                SA_TRY(Text::decode_tokens(
                    vocabulary, TokenSpan(generated.data(), generated.size()), text));

                print_inference_header(
                    run, SushiLog::format_text("{} tokens", generator->vocabulary()));
                std::cout << SushiLog::format_text("Generator  : {}\n", row.name)
                          << SushiLog::format_text("Sampler    : {}, seed {}\n",
                                                     task.sampler->name(), task.seed)
                          << SushiLog::format_text(
                                 "Generated  : {} tokens after a prompt of {}\n\n",
                                 generated.size(), prompt.size())
                          << text << "\n";
                return Status();
            }

        } // namespace

        Status register_language_domain(Registries& registries)
        {
            SA_TRY(Language::register_nodes(registries.nodes));
            SA_TRY(Language::register_datasets(registries.datasets));

            // The next-token objective fills its own leaf, so the fallback run trains it.
            SA_TRY(Language::register_objectives(registries.objectives));

            return registries.inferences.add(
                InferenceEntry{Generation::K_GENERATION_TASK_TYPE, &report_generation});
        }

    } // namespace Apps
} // namespace SushiAI
