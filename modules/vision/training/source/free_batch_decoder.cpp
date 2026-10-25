/**************************************************************************/
/* free_batch_decoder.cpp                                                 */
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
 * @file free_batch_decoder.cpp
 * @brief Implements the decode of one batch of anchor-free head outputs.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/free_batch_decoder.hpp>

#include <cmath>
#include <cstddef>

#include <SushiAI/failure/error.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Returns the logistic function of @p logit. */
        [[nodiscard]] float sigmoid(float logit)
        {
            return 1.0f / (1.0f + std::exp(-logit));
        }
    } // namespace

    FreeBatchDecoder::FreeBatchDecoder(const FreeModelFacts& facts, std::int64_t batch)
        : reg_max_(facts.reg_max), classes_(facts.classes), batch_(batch)
    {
        SA_THROW_IF(facts.levels.empty() || facts.reg_max < 1 || facts.classes < 1 || batch < 1,
                    "a decoder needs a level, a positive reg_max, classes and batch; got {} "
                    "levels, reg_max {}, {} classes, batch {}",
                    facts.levels.size(), facts.reg_max, facts.classes, batch);

        levels_.reserve(facts.levels.size());
        for (const FreeLevelFacts& level : facts.levels)
            levels_.push_back(Detect::Free::FeatureLevel{level.height, level.width,
                                                         static_cast<float>(level.stride)});

        anchors_ = Detect::Free::count_anchors(levels_);
        const auto anchors = static_cast<std::size_t>(anchors_);
        points_.resize(anchors * Detect::Free::K_POINT_VALUES);
        strides_.resize(anchors);
        Detect::Free::make_anchors(levels_, points_, strides_);

        const std::size_t rows = static_cast<std::size_t>(batch_) * anchors;
        bin_logits_.resize(rows * Detect::K_BOX_VALUES * static_cast<std::size_t>(reg_max_));
        distances_.resize(rows * Detect::K_BOX_VALUES);
    }

    std::int64_t FreeBatchDecoder::anchors() const noexcept
    {
        return anchors_;
    }

    void FreeBatchDecoder::decode(SushiRuntime::span<const FreeLevelLogits> levels,
                                  SushiRuntime::span<float> boxes,
                                  SushiRuntime::span<float> scores)
    {
        const auto rows = static_cast<std::size_t>(batch_ * anchors_);
        SA_THROW_IF(levels.size() != levels_.size() || boxes.size() != distances_.size()
                        || scores.size() != rows * static_cast<std::size_t>(classes_),
                    "a batch decode reads {} levels into [{}, {}, 4] boxes and [{}, {}, {}] "
                    "scores; got {} levels, {} box and {} score values",
                    levels_.size(), batch_, anchors_, batch_, anchors_, classes_,
                    levels.size(), boxes.size(), scores.size());

        const std::int64_t box_channels =
            static_cast<std::int64_t>(Detect::K_BOX_VALUES) * reg_max_;
        std::int64_t offset = 0;
        for (std::size_t k = 0; k < levels_.size(); ++k)
        {
            const std::int64_t level_anchors = levels_[k].height * levels_[k].width;
            Detect::Free::join_level(levels[k].boxes,
                                     Detect::Free::LevelPlacement{batch_, anchors_, box_channels,
                                                                  offset, level_anchors},
                                     bin_logits_);
            Detect::Free::join_level(levels[k].scores,
                                     Detect::Free::LevelPlacement{batch_, anchors_, classes_,
                                                                  offset, level_anchors},
                                     scores);
            offset += level_anchors;
        }

        Detect::Free::expect_distances(bin_logits_, reg_max_, distances_);
        Detect::Free::decode_xywh(distances_, points_, boxes);
        Detect::Free::scale_boxes(boxes, strides_, boxes);
        for (float& score : scores)
            score = sigmoid(score);
    }
} // namespace SushiAI::Vision::Training
