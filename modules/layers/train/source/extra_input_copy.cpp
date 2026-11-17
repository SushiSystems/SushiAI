/**************************************************************************/
/* extra_input_copy.cpp                                                   */
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
 * @file extra_input_copy.cpp
 * @brief Implements copy_extra_inputs.
 * @author Mustafa Garip
 */

#include <SushiAI/train/extra_input_copy.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/train/feed.hpp>

namespace SushiAI
{
    namespace Train
    {
        Status copy_extra_inputs(SushiRuntime::span<const NN::InputLeaf> leaves,
                                 const Data::Batch& batch, Graph::TensorPool& pool,
                                 SushiBLAS::BLASOps& blas, std::vector<NamedFeed>& feeds) noexcept
        {
            feeds.clear();
            SA_TRY(extra_feeds(leaves, batch, feeds));

            for (std::size_t at = 0; at < leaves.size(); ++at)
            {
                SushiBLAS::Tensor& destination = pool.tensor(leaves[at].value);
                SA_TRY(check_feed(*feeds[at].tensor, destination, leaves[at].name));
                blas.copy(*feeds[at].tensor, destination);
            }

            return Status();
        }

    } // namespace Train
} // namespace SushiAI
