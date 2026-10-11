/**************************************************************************/
/* run_progress.cpp                                                       */
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
 * @file run_progress.cpp
 * @brief Checks the seed of a resumed run against the one its checkpoint holds.
 * @author Mustafa Garip
 */

#include <SushiAI/io/run_progress.hpp>

namespace SushiAI
{
    namespace IO
    {
        Status check_run_seed(const RunProgress& saved, std::uint64_t seed, std::string_view file)
        {
            SA_FAIL_IF(saved.seed != seed, StatusCode::Incompatible,
                       "'{}' was written by a run with the seed {}, but this run states the "
                       "seed {}; a resumed run draws what the saved one would have drawn only "
                       "under the same seed",
                       file, saved.seed, seed);
            return Status();
        }

    } // namespace IO
} // namespace SushiAI
