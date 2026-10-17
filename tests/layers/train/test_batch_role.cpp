/**************************************************************************/
/* test_batch_role.cpp                                                    */
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
 * @file test_batch_role.cpp
 * @brief Tests the lookup of a role an objective reads from a batch.
 * @author Mustafa Garip
 */

#include <string>
#include <gtest/gtest.h>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/batch_role.hpp>

using namespace SushiAI;

namespace
{
    /** @brief Returns a batch holding a labels target and a boxes target, in that order. */
    Data::Batch labelled_batch()
    {
        Data::Batch batch{};
        batch.targets.push_back(Data::BatchTarget{std::string(Data::K_ROLE_LABELS), {}});
        batch.targets.push_back(Data::BatchTarget{std::string(Data::K_ROLE_BOXES), {}});
        return batch;
    }
} // namespace

/** @brief Tests that a role the batch holds is returned as a pointer into the batch. */
TEST(Small_BatchRole, AHeldRoleIsReturnedAsAPointerIntoTheBatch)
{
    const Data::Batch batch = labelled_batch();

    SA_ASSERT_OK_AND_ASSIGN(const Data::BatchTarget* const boxes,
                            Train::require_batch_role(batch, Data::K_ROLE_BOXES));
    EXPECT_EQ(boxes, &batch.targets[1]);
    EXPECT_EQ(boxes->role, Data::K_ROLE_BOXES);
}

/** @brief Tests that a role the batch lacks is refused with the role's name. */
TEST(Small_BatchRole, AMissingRoleIsRefusedByName)
{
    const Data::Batch batch = labelled_batch();

    SA_EXPECT_FAILURE(Train::require_batch_role(batch, Data::K_ROLE_MASK),
                      StatusCode::InvalidDescription, "'mask'", "objective");
    SA_EXPECT_FAILURE(Train::require_batch_role(Data::Batch{}, Data::K_ROLE_LABELS),
                      StatusCode::InvalidDescription, "'labels'");
}
