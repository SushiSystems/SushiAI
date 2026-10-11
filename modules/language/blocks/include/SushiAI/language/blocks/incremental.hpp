/**************************************************************************/
/* incremental.hpp                                                        */
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
 * @file incremental.hpp
 * @brief Names the leaves the incremental forms of the blocks are fed and the score a masked
 *        position takes.
 * @author Mustafa Garip
 */

#pragma once

#include <string_view>

namespace SushiAI
{
    namespace Language::Blocks
    {
        /** @brief Names the [1, 1] int32 leaf that holds the position of the slice a run reads. */
        inline constexpr std::string_view K_POSITION_LEAF = "position";

        /**
         * @brief Names the [1, T] leaf added to the scores of a run: 0 at each cache row that
         *        holds a position the fed one may read, K_MASKED_ATTENTION_SCORE at the others.
         */
        inline constexpr std::string_view K_ATTENTION_MASK_LEAF = "attention_mask";

        /** @brief Holds the score a position that may not be read takes before the softmax. */
        inline constexpr double K_MASKED_ATTENTION_SCORE = -1.0e9;

        /**
         * @brief Names the [1, 1] int32 leaf that holds the cache row a run of a rotary
         *        attention writes: the position of the token modulo T.
         */
        inline constexpr std::string_view K_CACHE_SLOT_LEAF = "cache_slot";

        /**
         * @brief Names the [1, pairs] leaf that holds the cosine of the angle of each pair at
         *        the position of the token a run reads, counted from the start of the text.
         */
        inline constexpr std::string_view K_ROTARY_COSINE_LEAF = "rotary_cosine";

        /** @brief Names the [1, pairs] leaf that holds the sines K_ROTARY_COSINE_LEAF pairs. */
        inline constexpr std::string_view K_ROTARY_SINE_LEAF = "rotary_sine";

    } // namespace Language::Blocks
} // namespace SushiAI
