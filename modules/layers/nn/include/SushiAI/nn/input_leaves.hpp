/**************************************************************************/
/* input_leaves.hpp                                                       */
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
 * @file input_leaves.hpp
 * @brief Declares the extra inputs of a model, the leaves a step declares for them and their list.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI
{
    namespace NN
    {
        /** @brief Describes one input a model reads beside its primary input. */
        struct ExtraInput
        {
            /** @brief Names the input to the rows that read it and to the batch role that feeds it. */
            std::string name;

            /** @brief Holds the shape of one sample, batch axis left out. */
            Shape sample;

            /** @brief Names the element type of the input. */
            DType dtype = DType::Float32;
        };

        /** @brief Names the leaf a step declared for one extra input. */
        struct InputLeaf
        {
            /** @brief Names the extra input the leaf holds. */
            std::string name;

            /** @brief Names the leaf in the traced graph. */
            Graph::ValueID value = Graph::K_NO_VALUE;
        };

        /** @brief Collects the leaves of the extra inputs of one step, in declaration order. */
        class InputLeafList final
        {
            public:
                /** @brief Appends @p leaf. */
                void add(InputLeaf leaf)
                {
                    leaves_.push_back(std::move(leaf));
                }

                /** @brief Returns the leaf declared for @p name, or K_NO_VALUE for none. */
                [[nodiscard]] Graph::ValueID find(std::string_view name) const noexcept
                {
                    for (const InputLeaf& leaf : leaves_)
                    {
                        if (leaf.name == name)
                            return leaf.value;
                    }

                    return Graph::K_NO_VALUE;
                }

                /**
                 * @brief Returns every leaf, in the order it was added.
                 * @return A view that the next add() invalidates.
                 */
                [[nodiscard]] SushiRuntime::span<const InputLeaf> leaves() const noexcept
                {
                    return SushiRuntime::span<const InputLeaf>(leaves_.data(), leaves_.size());
                }

            private:
                /** @brief Holds the leaves in the order they were added. */
                std::vector<InputLeaf> leaves_;
        };

    } // namespace NN
} // namespace SushiAI
