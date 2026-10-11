/**************************************************************************/
/* image_extensions.cpp                                                   */
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
 * @file image_extensions.cpp
 * @brief Implements the test of a path's extension against the image file extensions.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/datasets/image_extensions.hpp>

#include <string>
#include <string_view>

#include <SushiAI/vision/imaging/raw_cache.hpp>

namespace SushiAI::Vision::Datasets
{
    namespace
    {
        /** @brief Lists the extensions of an image file, in lower case. */
        constexpr std::string_view K_IMAGE_EXTENSIONS[] = {
            ".jpg", ".jpeg", ".png", Imaging::K_RAW_IMAGE_EXTENSION,
        };

        /** @brief Returns an ASCII capital as its small letter and any other byte as it is. */
        [[nodiscard]] constexpr char to_lower_ascii(char character) noexcept
        {
            return character >= 'A' && character <= 'Z'
                       ? static_cast<char>(character - 'A' + 'a')
                       : character;
        }
    } // namespace

    bool has_image_extension(const std::filesystem::path& path)
    {
        std::string extension = path.extension().generic_string();
        for (char& character : extension)
            character = to_lower_ascii(character);

        for (std::string_view known : K_IMAGE_EXTENSIONS)
        {
            if (extension == known)
                return true;
        }

        return false;
    }
} // namespace SushiAI::Vision::Datasets
