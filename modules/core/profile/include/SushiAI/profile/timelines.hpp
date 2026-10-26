/**************************************************************************/
/* timelines.hpp                                                          */
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
 * @file timelines.hpp
 * @brief Declares Timeline, mark_period and timeline_name, the periods of a training step and an inference batch.
 * @author Mustafa Garip
 * @see modules/core/profile/README.md
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace SushiAI
{
    namespace Profile
    {
        /** @brief Names a timeline whose periods the loops of SushiAI open. */
        enum class Timeline : std::uint8_t
        {
            TrainStep,      /**< "train.step": one period per optimizer step. */
            InferenceBatch, /**< "inference.batch": one period per batch run. */
        };

        /**
         * @brief Opens the next period of @p timeline when a session is bound; otherwise does
         *        nothing.
         */
        void mark_period(Timeline timeline) noexcept;

        /** @brief Returns the name of @p timeline, as a session's configuration names it. */
        [[nodiscard]] std::string_view timeline_name(Timeline timeline) noexcept;
    } // namespace Profile
} // namespace SushiAI
