/**************************************************************************/
/* anchor_bias_init.hpp                                                   */
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
 * @file anchor_bias_init.hpp
 * @brief Declares the stride-derived start of an anchor head's objectness and class biases.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the state entry of one head's last bias and the stride of its level. */
    struct AnchorBiasHead
    {
        std::string bias;         /**< State name of the head's last convolution bias. */
        std::int64_t stride = 0;  /**< Input pixels per cell, equal along H and W. */
    };

    /** @brief Returns the objectness bias log(8 / (640 / stride)^2). */
    [[nodiscard]] double anchor_objectness_bias(std::int64_t stride);

    /** @brief Returns the class bias log(0.6 / (classes - 0.99)). */
    [[nodiscard]] double anchor_class_bias(std::int64_t classes);

    /**
     * @brief Finds, per head of @p task, the bias entry it ends in and its level's stride.
     * @param spec The model file; a head ends in a "conv2d" row, alone or under a "sigmoid".
     * @param outputs The traced outputs; each head's output is looked up by its name.
     * @return The heads in task order, or InvalidDescription naming the output that is absent,
     *         not a rank-4 map, without a biased "conv2d" row, or of an unfit stride.
     */
    [[nodiscard]] Result<std::vector<AnchorBiasHead>> derive_anchor_bias_heads(
        const Config::ModelSpec& spec, const Shape& input_sample,
        const Detect::Anchor::AnchorTask& task, const std::vector<Train::OutputShape>& outputs);

    /**
     * @brief Adds the objectness and class biases to every anchor slot of every head, by
     *        writing the host storage.
     * @pre NN::initialize() has run; the entries it adds to keep the value that run gave them,
     *      which is zero for a bias, so a second call adds again.
     * @return InvalidDescription naming the entry that is absent, not float32, not of the
     *         channels the task gives, or without storage; @p state is unchanged then.
     */
    [[nodiscard]] Status initialize_anchor_biases(const Detect::Anchor::AnchorTask& task,
                                                  const std::vector<AnchorBiasHead>& heads,
                                                  NN::ModelState& state);

} // namespace SushiAI::Vision::Training
