/**************************************************************************/
/* model_writer.hpp                                                       */
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
 * @file model_writer.hpp
 * @brief Declares the writer that turns a model specification into the text of a model file.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/failure/result.hpp>

namespace SushiAI
{
    namespace Config
    {
        /**
         * @brief Writes @p model as a model file of version @ref K_MODEL_VERSION.
         * @param model The specification; an empty outputs list publishes the last row whole.
         * @return Text that parse_model reads back to an equal specification, or
         *         InvalidDescription for one no model file can state, or OutOfMemory.
         */
        [[nodiscard]] Result<std::string> write_model(const ModelSpec& model) noexcept;

    } // namespace Config
} // namespace SushiAI
