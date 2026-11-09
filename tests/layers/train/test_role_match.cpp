/**************************************************************************/
/* test_role_match.cpp                                                    */
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
 * @file test_role_match.cpp
 * @brief Tests match_roles: the datasets it accepts and the message of each refusal.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/data/in_memory_dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/role_match.hpp>

using namespace SushiAI;
using Data::InMemoryDataset;
using Data::TargetDescription;

namespace
{
    /** @brief Returns the labels role as one int32 per sample over @p classes classes. */
    TargetDescription labels_over(int64_t classes)
    {
        return TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                 classes};
    }

    /** @brief Returns a role named "boxes" of four float32 values per box. */
    TargetDescription boxes_of(int64_t boxes)
    {
        return TargetDescription{"boxes", Shape{boxes, 4}, DType::Float32, 0};
    }

    /** @brief Returns the labels role as one float32 per sample over 4 classes. */
    TargetDescription real_labels()
    {
        TargetDescription as_real = labels_over(4);
        as_real.dtype = DType::Float32;
        return as_real;
    }

    /** @brief Returns an in-memory dataset that publishes @p targets, with no batch. */
    InMemoryDataset publishing(std::vector<TargetDescription> targets)
    {
        return InMemoryDataset(0, 4, Data::InputDescription{Shape{6}, DType::Float32},
                               std::move(targets));
    }

    /** @brief Holds the roles a step requires and the roles a dataset publishes. */
    struct AcceptRow
    {
        std::string name;                           /**< The row's name. */
        std::vector<TargetDescription> required;    /**< The roles the step reads. */
        std::vector<TargetDescription> published;   /**< The roles the dataset publishes. */
    };

    /** @brief Matches the dataset of each accepting row. */
    class Small_RoleMatchAccepted : public ::testing::TestWithParam<AcceptRow>
    {

    };

    /** @brief Holds the roles of a refused match, its code and the texts its message holds. */
    struct RefuseRow
    {
        std::string name;                           /**< The row's name. */
        std::vector<TargetDescription> required;    /**< The roles the step reads. */
        std::vector<TargetDescription> published;   /**< The roles the dataset publishes. */
        StatusCode code;                            /**< The code of the refusal. */
        std::string first;                          /**< A text the message holds. */
        std::string second;                         /**< Another text the message holds. */
        std::string third;                          /**< A third text the message holds. */
    };

    /** @brief Matches the dataset of each refused row. */
    class Small_RoleMatchRefused : public ::testing::TestWithParam<RefuseRow>
    {

    };
} // namespace

/** @brief Tests that a dataset publishing the required roles, or none are required, is accepted. */
TEST_P(Small_RoleMatchAccepted, ADatasetThatPublishesTheRequiredRolesIsAccepted)
{
    const AcceptRow& row = GetParam();
    const InMemoryDataset dataset = publishing(row.published);

    SA_EXPECT_OK(Train::match_roles(row.required, dataset));
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RoleMatchAccepted,
    ::testing::Values(
        AcceptRow{"EveryRolePublished", {boxes_of(8), labels_over(4)},
                  {labels_over(4), boxes_of(8)}},
        AcceptRow{"NoRoleReadAndNonePublished", {}, {}},
        AcceptRow{"NoRoleReadAndLabelsPublished", {}, {labels_over(4)}},
        AcceptRow{"ARequiredClassCountOfZero", {labels_over(0)}, {labels_over(5)}}),
    [](const ::testing::TestParamInfo<AcceptRow>& info) { return info.param.name; });

/** @brief Tests that a role the dataset lacks or publishes differently is refused by name. */
TEST_P(Small_RoleMatchRefused, ARoleThatDoesNotMatchIsRefusedWithBothDescriptions)
{
    const RefuseRow& row = GetParam();
    const InMemoryDataset dataset = publishing(row.published);

    SA_EXPECT_FAILURE(Train::match_roles(row.required, dataset), row.code, "'in-memory'",
                      row.first, row.second, row.third);
}

INSTANTIATE_TEST_SUITE_P(
    Rows, Small_RoleMatchRefused,
    ::testing::Values(
        RefuseRow{"AMissingRoleNamesWhatIsPublished", {labels_over(4), boxes_of(8)},
                  {labels_over(4)}, StatusCode::InvalidDescription, "'boxes'",
                  "float32 [8, 4] per sample", "publishes 'labels'"},
        RefuseRow{"AMissingRoleOfADatasetThatPublishesNothing", {labels_over(4)}, {},
                  StatusCode::InvalidDescription, "'labels'", "(none)", "(none)"},
        RefuseRow{"AnotherShape", {boxes_of(8)}, {boxes_of(16)}, StatusCode::Incompatible,
                  "'boxes'", "as float32 [16, 4] per sample",
                  "compiled for float32 [8, 4] per sample"},
        RefuseRow{"AnotherElementType", {labels_over(4)}, {real_labels()},
                  StatusCode::Incompatible, "'labels'",
                  "as 4 classes of float32 [] per sample",
                  "compiled for 4 classes of int32 [] per sample"},
        RefuseRow{"AnotherClassCount", {labels_over(4)}, {labels_over(5)},
                  StatusCode::Incompatible, "'labels'", "as 5 classes of int32 [] per sample",
                  "compiled for 4 classes of int32 [] per sample"},
        RefuseRow{"TheFirstRoleThatDoesNotMatch", {boxes_of(8), labels_over(4)},
                  {labels_over(5)}, StatusCode::InvalidDescription, "'boxes'", "'boxes'",
                  "'boxes'"}),
    [](const ::testing::TestParamInfo<RefuseRow>& info) { return info.param.name; });
