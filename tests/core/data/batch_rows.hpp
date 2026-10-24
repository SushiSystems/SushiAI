/**************************************************************************/
/* batch_rows.hpp                                                         */
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
 * @file batch_rows.hpp
 * @brief Declares the check that a range read holds the rows of the whole batch.
 * @author Mustafa Garip
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>

namespace SushiAI
{
    /** @brief Holds what the tests of the data module share. */
    namespace DataTest
    {
        /**
         * @brief Returns whether @p part holds, byte for byte, rows @p rows of @p whole.
         * @param batch_rows The rows of @p whole.
         */
        inline ::testing::AssertionResult holds_rows_of(const Data::Batch& whole,
                                                        std::int64_t batch_rows,
                                                        Data::RowRange rows,
                                                        const Data::Batch& part)
        {
            if (SushiAI::Test::read_tensor_bytes(part.inputs)
                != SushiAI::Test::read_row_bytes(whole.inputs, batch_rows, rows.first,
                                                 rows.count))
                return ::testing::AssertionFailure() << "the inputs differ";

            if (part.targets.size() != whole.targets.size())
                return ::testing::AssertionFailure()
                       << part.targets.size() << " roles against " << whole.targets.size();

            for (std::size_t t = 0; t < whole.targets.size(); ++t)
            {
                if (part.targets[t].role != whole.targets[t].role
                    || SushiAI::Test::read_tensor_bytes(part.targets[t].values)
                           != SushiAI::Test::read_row_bytes(whole.targets[t].values, batch_rows,
                                                            rows.first, rows.count))
                    return ::testing::AssertionFailure()
                           << "the role '" << whole.targets[t].role << "' differs";
            }

            return ::testing::AssertionSuccess();
        }

    } // namespace DataTest
} // namespace SushiAI
