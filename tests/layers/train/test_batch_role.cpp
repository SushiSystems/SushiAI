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
 * @file test_batch_role.cpp
 * @brief Tests the lookup of a role an objective reads from a batch.
 * @author Mustafa Garip
 */

#include <string>
#include <string_view>
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

namespace
{
    /** @brief Holds a batch, the role asked for and the words of the refusal. */
    struct RefusalRow
    {
        std::string name;       /**< The row's name. */
        Data::Batch batch;      /**< The batch asked. */
        std::string_view role;  /**< The role asked for. */
        std::string first;      /**< A text the message holds. */
        std::string second;     /**< Another text the message holds. */
    };

    /** @brief Runs the refusal of a missing role over each batch. */
    class Small_BatchRoleRefusal : public ::testing::TestWithParam<RefusalRow>
    {
    };
} // namespace

/** @brief Tests that a role the batch lacks is refused with the role's name. */
TEST_P(Small_BatchRoleRefusal, AMissingRoleIsRefusedByName)
{
    const RefusalRow& row = GetParam();

    SA_EXPECT_FAILURE(Train::require_batch_role(row.batch, row.role),
                      StatusCode::InvalidDescription, row.first, row.second);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_BatchRoleRefusal,
    ::testing::Values(RefusalRow{"MaskOfALabelledBatch", labelled_batch(), Data::K_ROLE_MASK,
                                 "'mask'", "objective"},
                      RefusalRow{"LabelsOfAnEmptyBatch", Data::Batch{}, Data::K_ROLE_LABELS,
                                 "'labels'", "objective"}),
    [](const ::testing::TestParamInfo<RefusalRow>& info) { return info.param.name; });
