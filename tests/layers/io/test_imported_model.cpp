/**************************************************************************/
/* test_imported_model.cpp                                                */
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
 * @file test_imported_model.cpp
 * @brief Tests the empty state of ImportedModel and the payload size of an ImportedTensor.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/imported_model.hpp>

using namespace SushiAI;

/** @brief Tests that a default model states no weights, no training keys and no notes. */
TEST(Small_ImportedModel, ADefaultModelStatesNothing)
{
    const IO::ImportedModel imported;

    EXPECT_TRUE(imported.weights.empty());
    EXPECT_TRUE(imported.training.is_null());
    EXPECT_TRUE(imported.notes.empty());
    EXPECT_TRUE(imported.model.layers.empty());
    EXPECT_TRUE(imported.model.task.is_null());
}

/** @brief Tests that a tensor's payload holds the bytes its record states. */
TEST(Small_ImportedModel, ATensorHoldsTheBytesItsRecordStates)
{
    IO::ImportedTensor tensor;
    tensor.record.name = "0.weight";
    tensor.record.dtype = DType::Float32;
    tensor.record.shape = Shape{4, 3};
    tensor.payload.resize(tensor.record.payload_bytes());

    EXPECT_EQ(tensor.payload.size(), std::size_t{4 * 3 * sizeof(float)});

    tensor.record.dtype = DType::Int64;
    EXPECT_EQ(tensor.record.payload_bytes(), std::size_t{4 * 3 * 8});
}
