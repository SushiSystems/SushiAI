/**************************************************************************/
/* row_feed.hpp                                                           */
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
 * @file row_feed.hpp
 * @brief Declares RowFeed, which reads rows of a batch into the leaves of a training graph.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/execution/execution_backend.hpp>
#include <SushiAI/failure/status.hpp>

#include "training_setup.hpp"

namespace SushiAI
{
    namespace Apps
    {
        /**
         * @class RowFeed
         * @brief Reads rows of one batch of a setup's dataset into the leaves of its training
         *        graph.
         */
        class RowFeed final
        {
            public:
                /**
                 * @brief Borrows @p backend and @p setup.
                 * @param backend The backend of @p setup's engine; outlives the feed.
                 * @param setup Outlives the feed; its graph was traced over the rows fed.
                 */
                RowFeed(ExecutionBackend& backend, TrainingSetup& setup) noexcept;

                /**
                 * @brief Reads @p rows of batch @p batch_index and writes them into the leaves.
                 * @return The refusal of the read, of the objective's feed or of the input
                 *         copy.
                 * @pre The dataset entered the epoch the batch belongs to.
                 */
                [[nodiscard]] Status feed(std::size_t batch_index, Data::RowRange rows);

            private:
                /// @brief The backend the inputs are copied on; not owned.
                ExecutionBackend* backend_;

                /// @brief The data, the training graph and the pool; not owned.
                TrainingSetup* setup_;

                /// @brief The rows the last feed read; reused by the next one.
                Data::Batch part_;
        };
    } // namespace Apps
} // namespace SushiAI
