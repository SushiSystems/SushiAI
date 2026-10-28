/**************************************************************************/
/* free_objective_ports.hpp                                               */
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
 * @file free_objective_ports.hpp
 * @brief Declares the two-level model the anchor-free objective tests share.
 * @author Mustafa Garip
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>

/** @brief Holds the two-level model the anchor-free objective tests share. */
namespace SushiAI::FreeObjectiveTest
{
    /** @brief Holds the images per step. */
    inline constexpr std::int64_t K_BATCH = 2;

    /** @brief Holds the classes of the ports. */
    inline constexpr std::int64_t K_CLASSES = 2;

    /** @brief Holds the distribution bins per box side. */
    inline constexpr std::int64_t K_REG_MAX = 2;

    /** @brief Holds the side of the square input in pixels. */
    inline constexpr std::int64_t K_INPUT_SIDE = 32;

    /** @brief Holds the side of each level: 4 at stride 8, 2 at stride 16. */
    inline constexpr std::array<std::int64_t, 2> K_SIDES = {4, 2};

    /** @brief Returns the shape of the box (@p boxes) or the score port of @p level. */
    inline Shape port_shape(std::size_t level, bool boxes)
    {
        const std::int64_t side = K_SIDES[level];
        return Shape{K_BATCH, side, side, boxes ? 4 * K_REG_MAX : K_CLASSES};
    }

    /** @brief Returns the one-to-many ports of both levels, as a trace lists them. */
    inline std::vector<Train::OutputShape> list_ports()
    {
        using Vision::Detect::Free::detect_port_name;
        using Vision::Detect::Free::K_DETECT_BOXES;
        using Vision::Detect::Free::K_DETECT_ONE_TO_MANY;
        using Vision::Detect::Free::K_DETECT_SCORES;

        std::vector<Train::OutputShape> outputs;
        for (std::size_t level = 0; level < K_SIDES.size(); ++level)
        {
            outputs.push_back(Train::OutputShape{
                detect_port_name(K_DETECT_ONE_TO_MANY, level, K_DETECT_BOXES),
                port_shape(level, true)});
            outputs.push_back(Train::OutputShape{
                detect_port_name(K_DETECT_ONE_TO_MANY, level, K_DETECT_SCORES),
                port_shape(level, false)});
        }
        return outputs;
    }

    /** @brief Returns the context of a [32, 32, 3] model that publishes both levels. */
    inline Train::ObjectiveContext make_context()
    {
        Train::ObjectiveContext given;
        given.batch = K_BATCH;
        given.classes = K_CLASSES;
        given.input_sample = Shape{K_INPUT_SIDE, K_INPUT_SIDE, 3};
        given.task = Vision::Detect::Free::write_free_task(Vision::Detect::Free::FreeTask{})
                         .value();
        given.outputs = list_ports();
        return given;
    }
} // namespace SushiAI::FreeObjectiveTest
