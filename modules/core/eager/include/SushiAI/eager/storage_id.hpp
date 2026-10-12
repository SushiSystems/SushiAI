/**************************************************************************/
/* storage_id.hpp                                                         */
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
 * @file storage_id.hpp
 * @brief Declares StorageID, the name of one buffer a session keeps across steps.
 * @author Mustafa Garip
 * @see modules/core/eager/README.md
 */

#pragma once

#include <cstdint>

namespace SushiAI
{
    namespace Eager
    {
        /** @brief Names one buffer of a session; two values with one id share their bytes. */
        using StorageID = std::uint64_t;

        /** @brief Stands for "this value has no buffer of the session". */
        inline constexpr StorageID K_NO_STORAGE = 0;

        /**
         * @brief Hands out storage ids, each once.
         * @see modules/core/eager/README.md
         */
        class StorageIDSource final
        {
            public:
                /** @brief Returns an id no earlier call on this source returned; never 0. */
                [[nodiscard]] StorageID next() noexcept
                {
                    return ++last_;
                }

            private:
                /// @brief The id handed out last.
                StorageID last_ = K_NO_STORAGE;
        };

    } // namespace Eager
} // namespace SushiAI
