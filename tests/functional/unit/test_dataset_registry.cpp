/**************************************************************************/
/* test_dataset_registry.cpp                                              */
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
 * @file test_dataset_registry.cpp
 * @brief Tests the dataset registry: insertion order, lookup, refusal of a repeated type and
 *        the rows register_datasets adds.
 * @author Mustafa Garip
 */

#include <gtest/gtest.h>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include "../../common/test_status.hpp"

using namespace SushiAI;

namespace
{
    /** @brief The "rings" row, as register_datasets adds it. */
    constexpr Data::DatasetEntry K_RINGS_ROW{"rings", &Data::make_rings};

    /** @brief The "mnist" row, as register_mnist adds it. */
    constexpr Data::DatasetEntry K_MNIST_ROW{"mnist", &Data::make_mnist};
} // namespace

/** @brief Tests that rows() returns the rows in the order add() received them. */
TEST(Unit_DatasetRegistry, AddKeepsInsertionOrder)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(registry.add(K_RINGS_ROW));
    SA_EXPECT_OK(registry.add(K_MNIST_ROW));

    const SushiRuntime::span<const Data::DatasetEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].type, "rings");
    EXPECT_EQ(rows[1].type, "mnist");
}

/** @brief Tests that find returns null for a type no row holds and the row of one it holds. */
TEST(Unit_DatasetRegistry, FindReturnsNullForAnUnknownType)
{
    Data::DatasetRegistry registry;
    EXPECT_EQ(registry.find("rings"), nullptr);

    SA_EXPECT_OK(registry.add(K_RINGS_ROW));
    EXPECT_EQ(registry.find("coco"), nullptr);

    const Data::DatasetEntry* row = registry.find("rings");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->type, "rings");
}

/** @brief Tests that add refuses a type a row already holds and names the type. */
TEST(Unit_DatasetRegistry, ATypeAddedTwiceIsRefused)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(registry.add(K_RINGS_ROW));

    SA_EXPECT_FAILURE(registry.add(K_RINGS_ROW), StatusCode::InvalidDescription, "'rings'",
                      "already registered");
}

/** @brief Tests that register_datasets adds the "rings" row and no other. */
TEST(Unit_DatasetRegistry, RegisterDatasetsAddsRingsOnly)
{
    Data::DatasetRegistry registry;
    SA_EXPECT_OK(Data::register_datasets(registry));

    const SushiRuntime::span<const Data::DatasetEntry> rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].type, "rings");
    EXPECT_EQ(registry.find("mnist"), nullptr);
}
