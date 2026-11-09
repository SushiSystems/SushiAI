/**************************************************************************/
/* fed_leaves.hpp                                                        */
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
 * @file fed_leaves.hpp
 * @brief Declares FedLeaves, the seam that tells a position feed which leaves a step is fed.
 * @author Mustafa Garip
 */

#pragma once

#include <string>
#include <string_view>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>

namespace SushiAI::Language::Generation
{
    /** @brief Holds what a check of a leaf or of an output reads: its extents and its type. */
    struct LeafFacts
    {
        /** @brief Holds the extents of the leaf, outermost first. */
        Shape shape;

        /** @brief Holds the element type of the leaf. */
        DType dtype = DType::Float32;
    };

    /** @brief Names the leaves a one-position step is fed, with the shape and type of each. */
    class FedLeaves
    {
        public:
            /** @brief Releases the leaves through their base. */
            virtual ~FedLeaves() = default;

            /** @brief Lists the name of every leaf the step is fed. */
            [[nodiscard]] virtual SushiRuntime::span<const std::string> names()
                const noexcept = 0;

            /**
             * @brief Returns the shape and the type of the leaf @p name.
             * @return The facts, or InvalidDescription when the step has no such leaf.
             */
            [[nodiscard]] virtual Result<LeafFacts> leaf(std::string_view name)
                const noexcept = 0;
    };

} // namespace SushiAI::Language::Generation
