/**************************************************************************/
/* labels.hpp                                                             */
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
 * @file labels.hpp
 * @brief Declares the parser that turns one label file's text into a label set.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/imaging/label_set.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Imaging
{
    /** @brief Holds the field count of a box row: class, x centre, y centre, width, height. */
    inline constexpr std::int32_t K_BOX_ROW_FIELDS = 5;

    /** @brief Holds the smallest field count of a polygon row: class and three points. */
    inline constexpr std::int32_t K_POLYGON_ROW_MIN_FIELDS = 7;

    /** @brief Holds how far outside [0, 1] a coordinate may lie (data/utils.py:460-461). */
    inline constexpr float K_LABEL_COORDINATE_SLACK = 0.01f;

    /**
     * @brief Parses the text of one label file, throwing Error on anything the reference rejects.
     *
     * A file holds box rows only or polygon rows only. A polygon row gives the box around
     * its points and keeps them, in file order, as the object's polygon
     * (data/utils.py:446-450). README.md, section labels.hpp, lists every check.
     * @param class_count The dataset's class count; every class id must be below it.
     */
    [[nodiscard]] FileLabels parse_label_text(std::string_view text, std::int32_t class_count);
} // namespace SushiAI::Imaging
