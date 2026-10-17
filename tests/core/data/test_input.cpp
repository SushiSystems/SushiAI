/**************************************************************************/
/* test_input.cpp                                                         */
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
 * @file test_input.cpp
 * @brief Tests the description of the input a batch carries.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/data/input.hpp>

using namespace SushiAI;

/** @brief Tests that a description states float32 and a rank-0 sample unless told otherwise. */
TEST(Small_DataInput, ADefaultDescriptionIsAFloat32Scalar)
{
    const Data::InputDescription input;

    EXPECT_EQ(input.dtype, DType::Float32);
    EXPECT_EQ(input.sample_shape.rank(), 0);
}

/** @brief Tests that a description holds the shape and the element type it is given. */
TEST(Small_DataInput, ADescriptionHoldsWhatItIsGiven)
{
    const Data::InputDescription input{Shape{7, 2}, DType::Int32};

    EXPECT_EQ(input.dtype, DType::Int32);
    ASSERT_EQ(input.sample_shape.rank(), 2);
    EXPECT_EQ(input.sample_shape[0], 7);
    EXPECT_EQ(input.sample_shape[1], 2);
}
