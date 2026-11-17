/**************************************************************************/
/* extra_inputs.cpp                                                       */
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
 * @file extra_inputs.cpp
 * @brief Implements extra_feeds.
 * @author Mustafa Garip
 */

#include <SushiAI/train/extra_inputs.hpp>

#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/seam/seam.hpp>

namespace SushiAI
{
    namespace Train
    {
        Status extra_feeds(SushiRuntime::span<const NN::InputLeaf> leaves,
                           const Data::Batch& batch, std::vector<NamedFeed>& feeds) noexcept
        {
            SA_TRY(::SushiAI::Detail::guard(::SushiAI::Detail::SeamPhase::Allocate,
                                            [&]() -> Status
            {
                feeds.reserve(feeds.size() + leaves.size());
                return Status();
            }));

            for (const NN::InputLeaf& leaf : leaves)
            {
                const Data::BatchTarget* const role = Data::find_target(batch.targets, leaf.name);
                SA_FAIL_IF(role == nullptr, StatusCode::InvalidDescription,
                           "The batch holds no '{}' role, which feeds the extra input of that "
                           "name", leaf.name);

                feeds.push_back(NamedFeed{leaf.name, &role->values});
            }

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
