/**************************************************************************/
/* in_memory_sample_source.hpp                                            */
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
 * @file in_memory_sample_source.hpp
 * @brief Declares InMemorySampleSource, a SampleSource that holds its images in memory.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Supplies the images and labels it was given, by index; reads no file.
     *
     * load() changes no image and appends the index to a journal under a mutex, so several
     * threads may call it at once, as the interface allows. A source built from no image has
     * size() zero, which the interface rules out for a source a pipeline reads; add an image
     * before handing it to one.
     */
    class InMemorySampleSource final : public SampleSource
    {
        public:
            /**
             * @brief Appends one image and its labels.
             * @param image A decoded image of one channel or K_RGB_CHANNELS.
             * @param labels The labels of @p image, in fractions of its size.
             * @return The index the pair was given.
             */
            std::size_t add(Image image, FileLabels labels = {})
            {
                images_.push_back(std::move(image));
                labels_.push_back(std::move(labels));
                return images_.size() - 1;
            }

            /**
             * @brief Makes load() of @p index return @p failure instead of the image.
             * @param index In [0, size()); another index stops the process.
             * @param failure A failure; a success removes the scripted failure.
             */
            void fail_load_of(std::size_t index, Status failure)
            {
                SA_FATAL_IF(index >= images_.size(),
                            "InMemorySampleSource::fail_load_of: index {} is outside the {} images",
                            index, images_.size());

                if (failure.is_ok())
                    failures_.erase(index);
                else
                    failures_.insert_or_assign(index, std::move(failure));
            }

            /** @brief Returns the number of images added. */
            [[nodiscard]] std::size_t size() const noexcept override
            {
                return images_.size();
            }

            /**
             * @brief Returns the indices load() was called with, in call order.
             *
             * A call that failed, by a scripted failure, is recorded too; a call with an index
             * outside the source stops the process and is not.
             */
            [[nodiscard]] std::vector<std::size_t> loads() const
            {
                const std::lock_guard<std::mutex> lock(journal_mutex_);
                return journal_;
            }

            /**
             * @brief Copies image @p index and its labels into the outputs.
             * @return Success, or the failure scripted for @p index.
             * @pre @p index is in [0, size()); another index stops the process.
             */
            [[nodiscard]] Status load(std::size_t index, Image& image,
                                      FileLabels& labels) const override
            {
                SA_FATAL_IF(index >= images_.size(),
                            "InMemorySampleSource::load: index {} is outside the {} images",
                            index, images_.size());

                {
                    const std::lock_guard<std::mutex> lock(journal_mutex_);
                    journal_.push_back(index);
                }

                const auto scripted = failures_.find(index);
                if (scripted != failures_.end())
                    return scripted->second;

                image = images_[index];
                labels = labels_[index];
                return Status();
            }

        private:
            /// @brief Holds the images, by index.
            std::vector<Image> images_;

            /// @brief Holds the labels of each image, by index.
            std::vector<FileLabels> labels_;

            /// @brief Holds the failure scripted for an index.
            std::unordered_map<std::size_t, Status> failures_;

            /// @brief Guards journal_, which load() appends to from any thread.
            mutable std::mutex journal_mutex_;

            /// @brief Holds the indices load() was called with, in call order.
            mutable std::vector<std::size_t> journal_;
    };
} // namespace SushiAI::Vision::Imaging
