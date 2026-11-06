/**************************************************************************/
/* in_memory_sample_pipeline.hpp                                          */
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
 * @file in_memory_sample_pipeline.hpp
 * @brief Declares InMemorySamplePipeline, a SamplePipeline that loads no file.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include <SushiRuntime/core/span.hpp>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>

/** @brief Holds the readers of the on-disk layouts vision datasets are kept in. */
namespace SushiAI::Vision::Datasets
{
    /**
     * @brief Fills each fitted sample from the record of its sample in the list and loads no
     *        pixel.
     *
     * A fitted sample gets the file and the whole-image class of its record and keeps the image,
     * the objects, the mask and the tabular spans it came with. It keeps no state between two
     * calls, as the interface requires, so it journals nothing.
     */
    class InMemorySamplePipeline final : public SamplePipeline
    {
        public:
            /**
             * @brief Builds a pipeline that succeeds, or returns @p outcome when it is a failure.
             * @param outcome Returned by every produce() before any sample is touched.
             */
            explicit InMemorySamplePipeline(Status outcome = Status()) noexcept
                : outcome_(std::move(outcome))
            {

            }

            /** @brief Returns "in-memory". */
            [[nodiscard]] std::string_view name() const noexcept override
            {
                return "in-memory";
            }

            /**
             * @brief Copies the file and the class of each listed sample into its fitted sample.
             * @return Success; the scripted failure; or InvalidDescription for an index the
             *         list lacks, or for @p samples and @p fitted of different lengths.
             */
            [[nodiscard]] Status produce(const SampleList& list,
                                         SushiRuntime::span<const std::size_t> samples,
                                         std::int64_t,
                                         SushiRuntime::span<FittedSample> fitted) const override
            {
                if (!outcome_.is_ok())
                    return outcome_;

                SA_FAIL_IF(samples.size() != fitted.size(), StatusCode::InvalidDescription,
                           "InMemorySamplePipeline::produce: {} indices for {} fitted samples",
                           samples.size(), fitted.size());

                for (std::size_t row = 0; row < samples.size(); ++row)
                {
                    SA_FAIL_IF(samples[row] >= list.samples.size(),
                               StatusCode::InvalidDescription,
                               "InMemorySamplePipeline::produce: sample {} is outside the {} "
                               "of the list",
                               samples[row], list.samples.size());

                    const Sample& sample = list.samples[samples[row]];
                    fitted[row].file = sample.file;
                    fitted[row].class_index = sample.class_index;
                }

                return Status();
            }

        private:
            /// @brief Holds the status every produce() returns first.
            Status outcome_;
    };
} // namespace SushiAI::Vision::Datasets
