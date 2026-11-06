/**************************************************************************/
/* free_bias_init.hpp                                                     */
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
 * @file free_bias_init.hpp
 * @brief Declares the stride-derived start of an anchor-free head's last biases.
 * @author Mustafa Garip
 */

#pragma once

#include <cstdint>
#include <string_view>

#include <SushiAI/failure/status.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>

namespace SushiAI::Vision::Training
{
    /** @brief Returns the class bias log(5 / classes / (640 / stride)^2). */
    [[nodiscard]] double free_class_bias(std::int64_t classes, std::int64_t stride) noexcept;

    /**
     * @brief Sets the last box bias of every level to 1 and the last class bias to
     *        free_class_bias(), on each copy @p facts holds, by writing the host storage.
     * @param head_prefix The dotted name the detect row's state hangs under, such as "23".
     * @pre NN::initialize() has run; this overwrites only the biases it names.
     * @return InvalidDescription naming the entry that is absent, not float32, not of the
     *         channels @p facts gives, or without storage; @p state is unchanged then.
     */
    [[nodiscard]] Status initialize_free_biases(const FreeModelFacts& facts,
                                                std::string_view head_prefix,
                                                NN::ModelState& state) noexcept;

} // namespace SushiAI::Vision::Training
