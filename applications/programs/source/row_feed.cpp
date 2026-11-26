/**************************************************************************/
/* row_feed.cpp                                                           */
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
 * @file row_feed.cpp
 * @brief Implements RowFeed::feed(), the read, the objective's feed and the input copy.
 * @author Mustafa Garip
 */

#include "row_feed.hpp"

#include <vector>

#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/profile/profile.hpp>
#include <SushiAI/train/extra_input_copy.hpp>
#include <SushiAI/train/extra_inputs.hpp>
#include <SushiAI/train/leaf_writes.hpp>

namespace SushiAI
{
    namespace Apps
    {
        RowFeed::RowFeed(ExecutionBackend& backend, TrainingSetup& setup) noexcept
            : backend_(&backend), setup_(&setup)
        {
        }

        Status RowFeed::feed(std::size_t batch_index, Data::RowRange rows)
        {
            {
                SA_ZONE("data", "read");
                SA_TRY(setup_->dataset().read_rows(batch_index, rows, part_));
            }
            // No task is recorded here, so the objective may write its leaves on the host.
            const Train::TrainingGraph& built = setup_->built();
            {
                SA_ZONE("host", "feed");
                SA_TRY(Train::mark_leaves_unwritten(built.leaves, setup_->pool()));
                SA_TRY(setup_->objective().feed(part_, built.leaves, setup_->pool()));
                SA_TRY(Train::require_leaves_written(built.leaves, setup_->pool()));
            }
            SushiBLAS::Tensor& inputs = setup_->pool().tensor(built.spec.inputs);
            std::vector<Train::NamedFeed> extra_feeds;
            SA_ZONE("copy", "input");
            return SushiAI::Detail::record_and_run(*backend_,
                                                   [&](SushiBLAS::TaskRecorder& recorder)
            {
                SushiBLAS::BLASOps blas(recorder);
                blas.copy(part_.inputs, inputs);
                // The extra inputs of the model are read from the part as the trainer reads them.
                return Train::copy_extra_inputs(built.extra_inputs.leaves(), part_, setup_->pool(),
                                                blas, extra_feeds);
            });
        }
    } // namespace Apps
} // namespace SushiAI
