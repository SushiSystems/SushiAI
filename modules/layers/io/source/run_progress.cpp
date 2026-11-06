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
 * @file run_progress.cpp
 * @brief Checks the seed of a resumed run against the one its checkpoint holds.
 * @author Mustafa Garip
 */

#include <SushiAI/io/run_progress.hpp>

namespace SushiAI
{
    namespace IO
    {
        Status check_run_seed(const RunProgress& saved, std::uint64_t seed, std::string_view file) noexcept
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
