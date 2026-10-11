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
 * @file test_role_match.cpp
 * @brief Tests match_roles: the datasets it accepts and the message of each refusal.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/common.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/train/role_match.hpp>
#include <common/test_status.hpp>

using namespace SushiAI;
using Data::TargetDescription;

namespace
{
    /** @brief Publishes the roles it is built with and holds no batch. */
    class Published final : public Data::Dataset
    {
        public:
            /** @brief Holds @p targets as the roles the dataset publishes. */
            explicit Published(std::vector<TargetDescription> targets)
                : targets_(std::move(targets))
            {

            }

            int64_t batch_size() const noexcept override { return 4; }
            Shape sample_shape() const override { return Shape{6}; }
            const std::vector<TargetDescription>& targets() const noexcept override
            {
                return targets_;
            }
            std::size_t batch_count() const noexcept override { return 0; }
            std::string_view name() const noexcept override { return "published"; }

            [[nodiscard]] Status read(std::size_t index, Data::Batch&) const override
            {
                SA_FATAL_IF(true, "Batch {} is out of range; the dataset has 0", index);
                return Status();
            }

        private:
            /** @brief Holds the roles the dataset publishes. */
            std::vector<TargetDescription> targets_;
    };

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
} // namespace

/** @brief Tests that a dataset publishing every role as required is accepted. */
TEST(Unit_RoleMatch, ADatasetThatPublishesEveryRoleIsAccepted)
{
    const Published dataset({labels_over(4), boxes_of(8)});

    SA_EXPECT_OK(Train::match_roles({boxes_of(8), labels_over(4)}, dataset));
}

/** @brief Tests that a step that reads no role accepts a dataset that publishes none. */
TEST(Unit_RoleMatch, AStepThatReadsNoRoleAcceptsAnyDataset)
{
    const Published empty(std::vector<TargetDescription>{});
    const Published labelled({labels_over(4)});

    SA_EXPECT_OK(Train::match_roles({}, empty));
    SA_EXPECT_OK(Train::match_roles({}, labelled));
}

/** @brief Tests that a missing role is named with the dataset and the roles it publishes. */
TEST(Unit_RoleMatch, AMissingRoleIsNamedWithTheDatasetAndWhatItPublishes)
{
    const Published dataset({labels_over(4)});
    const Published empty(std::vector<TargetDescription>{});

    SA_EXPECT_FAILURE(Train::match_roles({labels_over(4), boxes_of(8)}, dataset),
                      StatusCode::InvalidDescription, "'published'", "'boxes'",
                      "float32 [8, 4] per sample", "publishes 'labels'");
    SA_EXPECT_FAILURE(Train::match_roles({labels_over(4)}, empty),
                      StatusCode::InvalidDescription, "'published'", "'labels'", "(none)");
}

/** @brief Tests that a role of another per-sample shape is refused with both shapes. */
TEST(Unit_RoleMatch, AnotherShapeIsRefusedWithBothShapes)
{
    const Published dataset({boxes_of(16)});

    SA_EXPECT_FAILURE(Train::match_roles({boxes_of(8)}, dataset), StatusCode::Incompatible,
                      "'published'", "'boxes'", "as float32 [16, 4] per sample",
                      "compiled for float32 [8, 4] per sample");
}

/** @brief Tests that a role of another element type is refused with both type names. */
TEST(Unit_RoleMatch, AnotherElementTypeIsRefusedWithBothTypes)
{
    TargetDescription as_real = labels_over(4);
    as_real.dtype = DType::Float32;
    const Published dataset({as_real});

    SA_EXPECT_FAILURE(Train::match_roles({labels_over(4)}, dataset), StatusCode::Incompatible,
                      "'published'", "'labels'", "as 4 classes of float32 [] per sample",
                      "compiled for 4 classes of int32 [] per sample");
}

/** @brief Tests that a role over another class count is refused with both counts. */
TEST(Unit_RoleMatch, AnotherClassCountIsRefusedWithBothCounts)
{
    const Published dataset({labels_over(5)});

    SA_EXPECT_FAILURE(Train::match_roles({labels_over(4)}, dataset), StatusCode::Incompatible,
                      "'published'", "'labels'", "as 5 classes of int32 [] per sample",
                      "compiled for 4 classes of int32 [] per sample");
}

/** @brief Tests that a required class count of 0 accepts whatever count is published. */
TEST(Unit_RoleMatch, ARequiredClassCountOfZeroAcceptsAnyCount)
{
    const Published dataset({labels_over(5)});

    SA_EXPECT_OK(Train::match_roles({labels_over(0)}, dataset));
}

/** @brief Tests that the first role that does not match is the one reported. */
TEST(Unit_RoleMatch, TheFirstRoleThatDoesNotMatchIsReported)
{
    const Published dataset({labels_over(5)});

    SA_EXPECT_FAILURE(Train::match_roles({boxes_of(8), labels_over(4)}, dataset),
                      StatusCode::InvalidDescription, "'boxes'");
}
