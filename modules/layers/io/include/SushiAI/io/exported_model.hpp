/**************************************************************************/
/* exported_model.hpp                                                     */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the Apache License, Version 2.0 (the "License"); you    */
/* may not use this file except in compliance with the License. You may   */
/* obtain a copy of the License at                                        */
/*                                                                        */
/*     https://www.apache.org/licenses/LICENSE-2.0                        */
/*                                                                        */
/* Unless required by applicable law or agreed to in writing, software    */
/* distributed under the License is distributed on an "AS IS" BASIS,      */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        */
/* implied. See the License for the specific language governing           */
/* permissions and limitations under the License.                         */
/**************************************************************************/

/**
 * @file exported_model.hpp
 * @brief Declares what an exporter returns: the files of a foreign format and the notes.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds one file of an export: its name in the output directory and its bytes. */
        struct ExportedFile
        {
            /** @brief Holds the file name, with no directory part. */
            std::string name;

            /** @brief Holds the whole file. */
            std::vector<std::byte> bytes;
        };

        /** @brief Holds a SushiAI model written as the files of another framework. */
        struct ExportedModel
        {
            /** @brief Holds the files in the order they are written. */
            std::vector<ExportedFile> files;

            /** @brief Lists what the model states that none of the files holds. */
            std::vector<std::string> notes;
        };

    } // namespace IO
} // namespace SushiAI
