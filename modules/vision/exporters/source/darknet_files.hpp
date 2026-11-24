/**************************************************************************/
/* darknet_files.hpp                                                      */
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
 * @file darknet_files.hpp
 * @brief Declares the writers of the .cfg text and the .weights bytes.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include "darknet_export_progress.hpp"

namespace SushiAI
{
    namespace Vision::Exporters
    {
        namespace Detail
        {
            /** @brief Returns the .cfg text: [net] of @p network, then @p sections in order. */
            [[nodiscard]] std::string write_darknet_cfg(
                const std::vector<DarknetOptionText>& network,
                const std::vector<DarknetSectionText>& sections);

            /**
             * @brief Returns the .weights bytes: major 0, minor 2, revision 0, a 64-bit seen of
             *        0, then each convolution's biases, scales, means, variances and weights.
             */
            [[nodiscard]] std::vector<std::byte> write_darknet_weights(
                const std::vector<DarknetConvolutionFloats>& convolutions);
        } // namespace Detail
    } // namespace Vision::Exporters
} // namespace SushiAI
