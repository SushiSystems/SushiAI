/**************************************************************************/
/* input_match.hpp                                                        */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2026-present Mustafa Garip & Sushi Systems               */
/*                                                                        */
/* Licensed under the PolyForm Noncommercial License 1.0.0 (the           */
/* "License"); you may not use this file except in compliance with the    */
/* License. You may obtain a copy of the License at                       */
/*                                                                        */
/*     https://polyformproject.org/licenses/noncommercial/1.0.0           */
/*                                                                        */
/* Noncommercial use is free. Commercial use requires a separate licence  */
/* from Sushi Systems; see COMMERCIAL.md. The software is provided        */
/* "as is", without warranty of any kind.                                 */
/**************************************************************************/

/**
 * @file input_match.hpp
 * @brief Declares the check of the input a step reads against the input a dataset publishes.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/graph/ir.hpp>

namespace SushiAI
{
    namespace Train
    {
        /**
         * @brief Returns the input a dataset must publish to feed @p leaf.
         * @param leaf The input leaf of a step, batch axis first.
         * @return The extents of @p leaf past its first and its element type, or
         *         InvalidDescription for a leaf of rank 0.
         */
        [[nodiscard]] Result<Data::InputDescription> describe_input_leaf(
            const Graph::Value& leaf) noexcept;

        /**
         * @brief Checks that @p dataset publishes its input as @p required describes it.
         * @param required What describe_input_leaf() returns for the input leaf of the step.
         * @return Success, or Incompatible naming @p dataset, and the element type and the
         *         per-sample shape on both sides.
         */
        [[nodiscard]] Status match_input(const Data::InputDescription& required,
                                         const Data::Dataset& dataset) noexcept;

    } // namespace Train
} // namespace SushiAI
