/**************************************************************************/
/* role_hidden_dataset.hpp                                                */
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
 * @file role_hidden_dataset.hpp
 * @brief Declares a dataset that serves another dataset's batches without one of its roles.
 * @author Mustafa Garip
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/shape/shape.hpp>

/** @brief Holds what the tests of the programs share to build a PreparedInference or a PreparedRun. */
namespace SushiAI::Apps::PreparedTest
{
    /**
     * @class RoleHiddenDataset
     * @brief Serves the batches of a dataset with one role left out of the roles and of every batch.
     */
    class RoleHiddenDataset final : public Data::Dataset
    {
        public:
            /**
             * @brief Wraps @p inner, hiding the target of @p hidden_role.
             * @param hidden_role A role that @p inner may or may not carry.
             */
            RoleHiddenDataset(std::unique_ptr<Data::Dataset> inner, std::string_view hidden_role)
                : inner_(std::move(inner)), hidden_role_(hidden_role)
            {
                for (const Data::TargetDescription& description : inner_->targets())
                {
                    if (description.role != hidden_role_)
                        targets_.push_back(description);
                }
            }

            /** @brief Returns the batch size of the wrapped dataset. */
            int64_t batch_size() const noexcept override
            {
                return inner_->batch_size();
            }

            /** @brief Returns the sample shape of the wrapped dataset. */
            Shape sample_shape() const override
            {
                return inner_->sample_shape();
            }

            /** @brief Returns the sample type of the wrapped dataset. */
            DType sample_dtype() const noexcept override
            {
                return inner_->sample_dtype();
            }

            /** @brief Returns the roles of the wrapped dataset but the hidden one. */
            const std::vector<Data::TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }

            /** @brief Returns the batch count of the wrapped dataset. */
            std::size_t batch_count() const noexcept override
            {
                return inner_->batch_count();
            }

            /** @brief Reads batch @p index of the wrapped dataset and drops the hidden role from it. */
            [[nodiscard]] Status read(std::size_t index, Data::Batch& batch) const override
            {
                SA_TRY(inner_->read(index, batch));
                hide(batch);
                return Status();
            }

            /** @brief Reads rows of the wrapped dataset and drops the hidden role from them. */
            [[nodiscard]] Status read_rows(std::size_t index, Data::RowRange rows,
                                           Data::Batch& batch) const override
            {
                SA_TRY(inner_->read_rows(index, rows, batch));
                hide(batch);
                return Status();
            }

            /** @brief Returns the name of the wrapped dataset. */
            std::string_view name() const noexcept override
            {
                return inner_->name();
            }

        private:
            /** @brief Removes the target of the hidden role from @p batch. */
            void hide(Data::Batch& batch) const
            {
                batch.targets.erase(std::remove_if(batch.targets.begin(), batch.targets.end(),
                                                   [this](const Data::BatchTarget& target)
                                                   {
                                                       return target.role == hidden_role_;
                                                   }),
                                    batch.targets.end());
            }

            /// @brief The dataset whose batches are served.
            std::unique_ptr<Data::Dataset> inner_;

            /// @brief The role that is left out.
            std::string hidden_role_;

            /// @brief The roles of the wrapped dataset but the hidden one.
            std::vector<Data::TargetDescription> targets_;
    };
} // namespace SushiAI::Apps::PreparedTest
