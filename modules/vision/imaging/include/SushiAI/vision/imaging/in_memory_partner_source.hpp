/**************************************************************************/
/* in_memory_partner_source.hpp                                           */
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
 * @file in_memory_partner_source.hpp
 * @brief Declares InMemoryPartnerSource, a PartnerSource that replays a scripted list.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <SushiAI/failure/check.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/random.hpp>

/** @brief Holds the host-side 8-bit image transforms of the detection data pipeline. */
namespace SushiAI::Vision::Imaging
{
    /**
     * @brief Returns the scripted partners in turn and journals the sample indices it is asked
     *        for; reads no file.
     *
     * The n-th call returns entry n modulo the script length. It takes three draws from the
     * generator and ignores their values, so a pipeline's draw sequence stays what the
     * interface promises.
     */
    class InMemoryPartnerSource final : public PartnerSource
    {
        public:
            /**
             * @brief Builds a source that replays @p script.
             * @param script The partners of each call, in call order; empty stops the process.
             */
            explicit InMemoryPartnerSource(std::vector<MosaicPartners> script)
                : script_(std::move(script))
            {
                SA_FATAL_IF(script_.empty(),
                            "InMemoryPartnerSource: the script holds no partners");
            }

            /** @brief Journals @p index, takes three draws and returns the next scripted entry. */
            [[nodiscard]] MosaicPartners choose(std::size_t index, Random& random) override
            {
                for (std::size_t draw = 0; draw < K_MOSAIC_PARTNERS; ++draw)
                    static_cast<void>(random.next_u32());

                asked_.push_back(index);
                return script_[(asked_.size() - 1) % script_.size()];
            }

            /** @brief Returns the sample index of each call, first call first. */
            [[nodiscard]] const std::vector<std::size_t>& asked() const noexcept
            {
                return asked_;
            }

        private:
            /// @brief Holds the partners to replay.
            std::vector<MosaicPartners> script_;

            /// @brief Holds the sample index of each call.
            std::vector<std::size_t> asked_;
    };
} // namespace SushiAI::Vision::Imaging
