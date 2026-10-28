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

#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/float_order.hpp>
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

    Result<FreeBatchDecoder> FreeBatchDecoder::make(const FreeModelFacts& facts,
                                                    std::int64_t batch)
    {
        SA_FAIL_IF(facts.levels.empty() || facts.reg_max < 1 || facts.classes < 1 || batch < 1,
                   StatusCode::InvalidDescription,
                   "a decoder needs a level, a positive reg_max, classes and batch; got {} "
                   "levels, reg_max {}, {} classes, batch {}",
                   facts.levels.size(), facts.reg_max, facts.classes, batch);

        return Result<FreeBatchDecoder>(FreeBatchDecoder(facts, batch));
    }

    FreeBatchDecoder::FreeBatchDecoder(const FreeModelFacts& facts, std::int64_t batch)
        : reg_max_(facts.reg_max), classes_(facts.classes), batch_(batch)
    {
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

    Status FreeBatchDecoder::decode(SushiRuntime::span<const FreeLevelLogits> levels,
                                    SushiRuntime::span<float> boxes,
                                    SushiRuntime::span<float> scores)
    {
        const auto rows = static_cast<std::size_t>(batch_ * anchors_);
        SA_FAIL_IF(levels.size() != levels_.size() || boxes.size() != distances_.size()
                       || scores.size() != rows * static_cast<std::size_t>(classes_),
                   StatusCode::InvalidDescription,
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
            const std::int64_t bad_box = Detect::first_non_finite(levels[k].boxes);
            SA_FAIL_IF(bad_box >= 0, StatusCode::InvalidDescription,
                       "level {} box output {} is {}; every output must be finite",
                       k, bad_box, levels[k].boxes[static_cast<std::size_t>(bad_box)]);
            const std::int64_t bad_score = Detect::first_non_finite(levels[k].scores);
            SA_FAIL_IF(bad_score >= 0, StatusCode::InvalidDescription,
                       "level {} score {} is {}; every score must be finite", k,
                       bad_score, levels[k].scores[static_cast<std::size_t>(bad_score)]);
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

        SA_TRY(Detect::Free::expect_distances(bin_logits_, reg_max_, distances_));
        Detect::Free::decode_xywh(distances_, points_, boxes);
        Detect::Free::scale_boxes(boxes, strides_, boxes);
        for (float& score : scores)
            score = sigmoid(score);

        return Status();
    }
} // namespace SushiAI::Vision::Training
