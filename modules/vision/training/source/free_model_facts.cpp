/**************************************************************************/
/* free_model_facts.cpp                                                   */
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
 * @file free_model_facts.cpp
 * @brief Implements the derivation of an anchor-free model's facts from its traced outputs.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_model_facts.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        using Detect::Free::detect_port_name;
        using Detect::Free::K_DETECT_BOXES;
        using Detect::Free::K_DETECT_ONE_TO_MANY;
        using Detect::Free::K_DETECT_ONE_TO_ONE;
        using Detect::Free::K_DETECT_SCORES;

        /** @brief Holds the rank of one input sample, [H, W, C]. */
        constexpr std::int32_t K_SAMPLE_RANK = 3;

        /** @brief Holds the rank of a published head output, [B, H, W, channels]. */
        constexpr std::int32_t K_PORT_RANK = 4;

        /** @brief Holds the box sides each position predicts a distribution for. */
        constexpr std::int64_t K_BOX_SIDES = 4;

        /** @brief Holds what derivation keeps while it walks the levels. */
        class FactsBuilder
        {
            public:
                /** @brief Binds the input sample and the outputs the facts are read from. */
                FactsBuilder(const Shape& input_sample,
                             const std::vector<Train::OutputShape>& outputs) noexcept
                    : input_(input_sample), outputs_(outputs)
                {

                }

                /** @brief Reports whether the model publishes level 0 of @p copy. */
                [[nodiscard]] bool publishes(std::string_view copy) const
                {
                    return Train::find_output_shape(
                               outputs_, detect_port_name(copy, 0, K_DETECT_BOXES))
                           != nullptr;
                }

                /** @brief Reports whether the one-to-many copy publishes @p level. */
                [[nodiscard]] bool has_level(std::size_t level) const
                {
                    return Train::find_output_shape(
                               outputs_,
                               detect_port_name(K_DETECT_ONE_TO_MANY, level, K_DETECT_BOXES))
                           != nullptr;
                }

                /** @brief Reads level @p level of @p copy into @p ports and @p level_facts. */
                [[nodiscard]] Status read_copy(std::string_view copy, std::size_t level,
                                               FreeLevelFacts& level_facts, FreeLevelPorts& ports)
                {
                    ports.boxes = detect_port_name(copy, level, K_DETECT_BOXES);
                    ports.scores = detect_port_name(copy, level, K_DETECT_SCORES);

                    SA_TRY_ASSIGN(const Shape* const boxes, find(ports.boxes));
                    SA_TRY_ASSIGN(const Shape* const scores, find(ports.scores));
                    SA_FAIL_IF(boxes->dim(1) != scores->dim(1) || boxes->dim(2) != scores->dim(2),
                               StatusCode::InvalidDescription,
                               "the output '{}' is {}, but '{}' is {}; a level's ports share H "
                               "and W", ports.boxes, boxes->to_string(), ports.scores,
                               scores->to_string());
                    SA_FAIL_IF(boxes->dim(3) % K_BOX_SIDES != 0, StatusCode::InvalidDescription,
                               "the output '{}' has {} channels, which is no multiple of {}",
                               ports.boxes, boxes->dim(3), K_BOX_SIDES);

                    SA_TRY(agree("reg_max", ports.boxes, boxes->dim(3) / K_BOX_SIDES,
                                 facts_.reg_max));
                    SA_TRY(agree("classes", ports.scores, scores->dim(3), facts_.classes));
                    if (level_facts.height == 0)
                        return derive_stride(ports.boxes, *boxes, level_facts);

                    SA_FAIL_IF(boxes->dim(1) != level_facts.height
                                   || boxes->dim(2) != level_facts.width,
                               StatusCode::InvalidDescription,
                               "the output '{}' is {}, but the level's other copy is {} by {}",
                               ports.boxes, boxes->to_string(), level_facts.height,
                               level_facts.width);
                    return Status();
                }

                /** @brief Moves the facts out. */
                [[nodiscard]] FreeModelFacts take() noexcept
                {
                    return std::move(facts_);
                }

                /** @brief Returns the facts being built. */
                [[nodiscard]] FreeModelFacts& facts() noexcept
                {
                    return facts_;
                }

            private:
                /** @brief Returns the shape of output @p name, which must be of rank 4. */
                [[nodiscard]] Result<const Shape*> find(const std::string& name) const
                {
                    const Train::OutputShape* const output =
                        Train::find_output_shape(outputs_, name);
                    SA_FAIL_IF(output == nullptr, StatusCode::InvalidDescription,
                               "the model does not publish the output '{}'; an anchor-free "
                               "model publishes the detect row's ports under their own names",
                               name);
                    SA_FAIL_IF(output->shape.rank() != K_PORT_RANK,
                               StatusCode::InvalidDescription,
                               "the output '{}' is {}; a detect port is [B, H, W, channels]", name,
                               output->shape.to_string());
                    return &output->shape;
                }

                /** @brief Sets @p held to @p value the first time; later values must equal it. */
                [[nodiscard]] static Status agree(std::string_view what, const std::string& name,
                                                  std::int64_t value, std::int64_t& held)
                {
                    SA_FAIL_IF(value < 1, StatusCode::InvalidDescription,
                               "the output '{}' gives {} {}; it must be at least 1", name, what,
                               value);
                    SA_FAIL_IF(held != 0 && held != value, StatusCode::InvalidDescription,
                               "the output '{}' gives {} {}, but an earlier port gives {}", name,
                               what, value, held);
                    held = value;
                    return Status();
                }

                /** @brief Sets the extents and the stride of @p level_facts from @p shape. */
                [[nodiscard]] Status derive_stride(const std::string& name, const Shape& shape,
                                                   FreeLevelFacts& level_facts) const
                {
                    const std::int64_t height = shape.dim(1);
                    const std::int64_t width = shape.dim(2);
                    SA_FAIL_IF(height < 1 || width < 1 || input_.dim(0) % height != 0
                                   || input_.dim(1) % width != 0
                                   || input_.dim(0) / height != input_.dim(1) / width,
                               StatusCode::InvalidDescription,
                               "the output '{}' is {} by {} over an input of {} by {}; a level's "
                               "stride must divide the input exactly and be equal along H and W",
                               name, height, width, input_.dim(0), input_.dim(1));

                    level_facts.height = height;
                    level_facts.width = width;
                    level_facts.stride = input_.dim(0) / height;
                    return Status();
                }

                /** @brief Holds the [H, W, C] input sample. */
                const Shape& input_;

                /** @brief Holds the published outputs. */
                const std::vector<Train::OutputShape>& outputs_;

                /** @brief Holds the facts read so far. */
                FreeModelFacts facts_;
        };
    } // namespace

    Result<FreeModelFacts> derive_free_model_facts(const Shape& input_sample,
                                                   const std::vector<Train::OutputShape>& outputs) noexcept
    {
        SA_FAIL_IF(input_sample.rank() != K_SAMPLE_RANK, StatusCode::InvalidDescription,
                   "the model reads samples of {}; an anchor-free model needs [H, W, C]",
                   input_sample.to_string());

        FactsBuilder builder(input_sample, outputs);
        SA_FAIL_IF(!builder.publishes(K_DETECT_ONE_TO_MANY), StatusCode::InvalidDescription,
                   "the model does not publish the output '{}'; an anchor-free model publishes "
                   "the detect row's ports under their own names",
                   detect_port_name(K_DETECT_ONE_TO_MANY, 0, K_DETECT_BOXES));

        FreeModelFacts& facts = builder.facts();
        facts.has_one_to_one = builder.publishes(K_DETECT_ONE_TO_ONE);
        for (std::size_t level = 0; builder.has_level(level); ++level)
        {
            FreeLevelFacts level_facts;
            SA_TRY(builder.read_copy(K_DETECT_ONE_TO_MANY, level, level_facts,
                                     level_facts.one_to_many));
            if (facts.has_one_to_one)
            {
                SA_TRY(builder.read_copy(K_DETECT_ONE_TO_ONE, level, level_facts,
                                         level_facts.one_to_one));
            }
            facts.levels.push_back(std::move(level_facts));
        }

        return Result<FreeModelFacts>(builder.take());
    }

} // namespace SushiAI::Vision::Training
