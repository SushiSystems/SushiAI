/**************************************************************************/
/* in_memory_collator.hpp                                                 */
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
 * @file in_memory_collator.hpp
 * @brief Declares InMemoryCollator, a Collator that writes no tensor.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Names a task, publishes the roles it was given and writes no element of a batch.
     *
     * collate() returns the outcome the collator was built with and touches neither the batch nor
     * a sample, so a test reads how a caller treats a collator without an engine or an image.
     * The adapter may count its own end, which shows that destroying through the interface
     * reached it.
     */
    class InMemoryCollator final : public Collator
    {
        public:
            /**
             * @brief Builds a collator of the task @p task over the roles @p targets.
             * @param task The name task() returns.
             * @param targets The roles targets() returns.
             * @param outcome What every collate() returns.
             * @param ended When not null, raised by one when the collator is destroyed; must
             *              outlive the collator.
             */
            InMemoryCollator(std::string task, std::vector<Data::TargetDescription> targets,
                             Status outcome = Status(), std::size_t* ended = nullptr)
                : task_(std::move(task)), targets_(std::move(targets)),
                  outcome_(std::move(outcome)), ended_(ended)
            {

            }

            InMemoryCollator(const InMemoryCollator&) = delete;
            InMemoryCollator& operator=(const InMemoryCollator&) = delete;
            InMemoryCollator(InMemoryCollator&&) = delete;
            InMemoryCollator& operator=(InMemoryCollator&&) = delete;

            /** @brief Raises the counter it was given, if any. */
            ~InMemoryCollator() override
            {
                if (ended_ != nullptr)
                    ++*ended_;
            }

            /** @brief Returns the task it was built with. */
            [[nodiscard]] std::string_view task() const noexcept override
            {
                return task_;
            }

            /** @brief Returns the roles it was built with. */
            [[nodiscard]] const std::vector<Data::TargetDescription>& targets()
                const noexcept override
            {
                return targets_;
            }

            /** @brief Returns the outcome it was built with and writes nothing. */
            [[nodiscard]] Status collate(const SampleList&, SushiRuntime::span<const FittedSample>,
                                         Data::Batch&) const override
            {
                return outcome_;
            }

        private:
            /// @brief Holds the name task() returns.
            std::string task_;

            /// @brief Holds the roles targets() returns.
            std::vector<Data::TargetDescription> targets_;

            /// @brief Holds the status every collate() returns.
            Status outcome_;

            /// @brief Holds the counter the destructor raises, or null.
            std::size_t* ended_;
    };
} // namespace SushiAI::Vision::Datasets
