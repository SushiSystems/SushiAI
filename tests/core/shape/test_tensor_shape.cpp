/**************************************************************************/
/* test_tensor_shape.cpp                                                  */
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
 * @file test_tensor_shape.cpp
 * @brief Tests that shape_of reads the extents of a SushiBLAS tensor as a Shape.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/error.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/failure_expectations/throw_expectations.hpp>

using SushiAI::Shape;

namespace
{
    /** @brief Returns a tensor description with no storage whose extents are @p a, @p b, @p c. */
    SushiBLAS::Tensor description(std::int64_t a, std::int64_t b, std::int64_t c)
    {
        SushiBLAS::Tensor tensor;
        tensor.rank = 3;
        tensor.shape[0] = a;
        tensor.shape[1] = b;
        tensor.shape[2] = c;
        return tensor;
    }
} // namespace

/** @brief Tests that the extents come back outermost first, up to the tensor's rank only. */
TEST(Small_TensorShape, ExtentsComeBackOutermostFirstUpToTheRank)
{
    SushiBLAS::Tensor tensor = description(2, 3, 5);
    tensor.shape[3] = 7;

    EXPECT_EQ(SushiAI::shape_of(tensor), (Shape{2, 3, 5}));
}

/** @brief Tests that a tensor of rank zero reads as the scalar Shape. */
TEST(Small_TensorShape, ARankZeroTensorIsTheScalarShape)
{
    const SushiBLAS::Tensor tensor;

    EXPECT_EQ(SushiAI::shape_of(tensor), Shape());
}

/** @brief Tests that an extent that is not positive throws an Error naming it. */
TEST(Small_TensorShape, AnExtentThatIsNotPositiveThrows)
{
    const SushiBLAS::Tensor tensor = description(2, 0, 5);

    SA_EXPECT_ERROR((void)SushiAI::shape_of(tensor), "positive", "0");
}
