/**************************************************************************/
/* imported_model.hpp                                                     */
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
 * @file imported_model.hpp
 * @brief Declares what an importer returns: a model, its weights and its training recipe.
 * @author Mustafa Garip
 * @see docs/design/MODEL_IMPORT/README.md
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/io/checkpoint_format.hpp>

namespace SushiAI
{
    namespace IO
    {
        /** @brief Holds one tensor of an imported model: its record and its bytes on the host. */
        struct ImportedTensor
        {
            /** @brief Holds the name, the element type and the extents the weight file states. */
            TensorRecord record;

            /** @brief Holds record.payload_bytes() bytes, row-major, in the host's byte order. */
            std::vector<std::byte> payload;
        };

        /** @brief Holds a model of another framework as the three files of SushiAI state it. */
        struct ImportedModel
        {
            /** @brief Holds the rows, the published outputs and the task section. */
            Config::ModelSpec model;

            /** @brief Holds every parameter and buffer by name; empty when no weights came. */
            std::vector<ImportedTensor> weights;

            /// @brief Holds the keys of a training file the foreign files state, as an object:
            ///        "objective", "optimizer", "schedule", "epochs". Null states none.
            Config::Value training;

            /** @brief Lists what the foreign files state that none of the three files holds. */
            std::vector<std::string> notes;
        };

    } // namespace IO
} // namespace SushiAI
