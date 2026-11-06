/**************************************************************************/
/* in_memory_mask_source.hpp                                              */
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
 * @file in_memory_mask_source.hpp
 * @brief Declares InMemoryMaskSource, a MaskSource that holds its masks in memory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Supplies the masks it was given, by index; reads no file.
     *
     * The index of a mask is the index of its image in the SampleSource it pairs with. load()
     * changes nothing, so several threads may call it at once.
     */
    class InMemoryMaskSource final : public MaskSource
    {
        public:
            /**
             * @brief Appends one mask.
             * @param mask One channel, one class index per pixel, the size of its image.
             * @return The index the mask was given.
             */
            std::size_t add(Image mask)
            {
                masks_.push_back(std::move(mask));
                return masks_.size() - 1;
            }

            /**
             * @brief Makes load() of @p index return @p failure instead of the mask.
             * @param index In [0, count()); another index stops the process.
             * @param failure A failure; a success removes the scripted failure.
             */
            void fail_load_of(std::size_t index, Status failure)
            {
                SA_FATAL_IF(index >= masks_.size(),
                            "InMemoryMaskSource::fail_load_of: index {} is outside the {} masks",
                            index, masks_.size());

                if (failure.is_ok())
                    failures_.erase(index);
                else
                    failures_.insert_or_assign(index, std::move(failure));
            }

            /** @brief Returns the number of masks added. */
            [[nodiscard]] std::size_t count() const noexcept
            {
                return masks_.size();
            }

            /**
             * @brief Copies mask @p index into @p mask.
             * @return Success, or the failure scripted for @p index.
             * @pre @p index is in [0, count()); another index stops the process.
             */
            [[nodiscard]] Status load(std::size_t index, Image& mask) const override
            {
                SA_FATAL_IF(index >= masks_.size(),
                            "InMemoryMaskSource::load: index {} is outside the {} masks",
                            index, masks_.size());

                const auto scripted = failures_.find(index);
                if (scripted != failures_.end())
                    return scripted->second;

                mask = masks_[index];
                return Status();
            }

        private:
            /// @brief Holds the masks, by index.
            std::vector<Image> masks_;

            /// @brief Holds the failure scripted for an index.
            std::unordered_map<std::size_t, Status> failures_;
    };
} // namespace SushiAI::Vision::Imaging
