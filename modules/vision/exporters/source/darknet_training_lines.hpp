/**************************************************************************/
/* darknet_training_lines.hpp                                             */
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
 * @file darknet_training_lines.hpp
 * @brief Declares the writing of a training file and a data block as the lines of a .cfg.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <vector>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/io/imported_model.hpp>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Holds the training lines of a .cfg and the notes for what it leaves out. */
            struct DarknetTrainingLines
            {
                std::vector<DarknetOptionText> network; /**< The lines of [net]. */
                std::vector<DarknetOptionText> head;    /**< The lines of every [yolo]. */
                std::vector<std::string> notes;         /**< What no .cfg line states. */
            };

            /**
             * @brief Returns the solver lines of [net] and the training lines of [yolo] that
             *        the training recipe and the data block of @p model state.
             * @return The lines and a note for each value left out; InvalidDescription for an
             *         objective or a schedule section that does not read.
             */
            [[nodiscard]] Result<DarknetTrainingLines> describe_darknet_training(
                const IO::ImportedModel& model);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
