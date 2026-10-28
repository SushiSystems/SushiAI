/**************************************************************************/
/* free_model_facts.hpp                                                   */
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
 * @file free_model_facts.hpp
 * @brief Declares the facts of an anchor-free model that its traced outputs state.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Holds the published names of one copy's box and class outputs of one level. */
    struct FreeLevelPorts
    {
        std::string boxes;  /**< Output of 4 * reg_max box channels. */
        std::string scores; /**< Output of one logit per class. */
    };

    /** @brief Holds what the trace says about one level, finest first. */
    struct FreeLevelFacts
    {
        std::int64_t height = 0;     /**< Positions along H. */
        std::int64_t width = 0;      /**< Positions along W. */
        std::int64_t stride = 0;     /**< Input pixels per position, equal along H and W. */
        FreeLevelPorts one_to_many;  /**< Ports of the one-to-many copy. */
        FreeLevelPorts one_to_one;   /**< Ports of the one-to-one copy; empty without it. */
    };

    /** @brief Holds what an anchor-free model's traced outputs state. */
    struct FreeModelFacts
    {
        std::int64_t reg_max = 0;           /**< Distribution bins per box side. */
        std::int64_t classes = 0;           /**< Class logits per position. */
        bool has_one_to_one = false;        /**< Whether the one-to-one copy is published. */
        std::vector<FreeLevelFacts> levels; /**< One entry per level, finest first. */
    };

    /**
     * @brief Derives the levels, ports, strides, reg_max and classes from a traced model.
     * @param input_sample The [H, W, C] shape of one input sample.
     * @param outputs The published outputs; the detect row's ports under their own names.
     * @return The facts, or InvalidDescription naming the output or the level that does not
     *         fit: a missing port, a rank other than 4, a stride that does not divide the
     *         input or differs between H and W, or channels that differ between levels.
     */
    [[nodiscard]] Result<FreeModelFacts> derive_free_model_facts(
        const Shape& input_sample, const std::vector<Train::OutputShape>& outputs) noexcept;

} // namespace SushiAI::Vision::Training
