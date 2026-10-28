/**************************************************************************/
/* image_extensions.hpp                                                   */
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
 * @file image_extensions.hpp
 * @brief Declares the file name extensions a reader takes for an image file.
 * @author Mustafa Garip
 */

#pragma once

#include <filesystem>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Reports whether a path's extension names a format vision/codecs decodes.
     *
     * The extensions are .jpg, .jpeg, .png and Imaging::K_RAW_IMAGE_EXTENSION, compared
     * without regard to the case of ASCII letters. The file is not opened.
     */
    [[nodiscard]] bool has_image_extension(const std::filesystem::path& path) noexcept;
} // namespace SushiAI::Vision::Datasets
