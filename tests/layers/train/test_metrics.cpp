/**************************************************************************/
/* test_metrics.cpp                                                       */
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
 * @file test_metrics.cpp
 * @brief Tests the named numbers of a step and the lookup of a step's one output.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/step_view.hpp>

using namespace SushiAI;
using Train::Metric;
using Train::Metrics;

namespace
{
    /** @brief Holds one more name than a Metrics value keeps. */
    constexpr const char* K_NAMES[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i"};

    /** @brief Returns the view of a step over @p batch that published @p outputs. */
    Train::StepView view_of(const Data::Batch& batch,
                            const std::vector<Train::StepOutput>& outputs)
    {
        return Train::StepView{
            batch, 0.0,
            SushiRuntime::span<const Train::StepOutput>(outputs.data(), outputs.size())};
    }
} // namespace

/** @brief Tests that metrics are kept in the order they were added and found by name. */
TEST(Unit_Metrics, MetricsAreKeptInOrderAndFoundByName)
{
    Metrics metrics;
    EXPECT_TRUE(metrics.all().empty());

    SA_EXPECT_OK(metrics.add(Metric{"accuracy", 0.75, true}));
    SA_EXPECT_OK(metrics.add(Metric{"perplexity", 12.5, false}));

    ASSERT_EQ(metrics.all().size(), 2u);
    EXPECT_EQ(metrics.all()[0].name, "accuracy");
    EXPECT_EQ(metrics.all()[1].name, "perplexity");
    ASSERT_NE(metrics.find("perplexity"), nullptr);
    EXPECT_EQ(metrics.find("perplexity")->value, 12.5);
    EXPECT_EQ(metrics.find("recall"), nullptr);
    EXPECT_EQ(metrics.value_or("accuracy", -1.0), 0.75);
    EXPECT_EQ(metrics.value_or("recall", -1.0), -1.0);

    metrics.clear();
    EXPECT_TRUE(metrics.all().empty());
    EXPECT_EQ(metrics.find("accuracy"), nullptr);
}

/** @brief Tests that an empty name, a second metric of one name and a ninth are refused. */
TEST(Unit_Metrics, AnEmptyNameADuplicateAndAnOverflowAreRefused)
{
    Metrics metrics;
    SA_EXPECT_FAILURE(metrics.add(Metric{"", 1.0, false}), StatusCode::InvalidDescription,
                      "non-empty name");

    SA_EXPECT_OK(metrics.add(Metric{"a", 1.0, false}));
    SA_EXPECT_FAILURE(metrics.add(Metric{"a", 2.0, false}), StatusCode::InvalidDescription,
                      "'a'", "twice");
    EXPECT_EQ(metrics.find("a")->value, 1.0);

    for (std::size_t index = 1; index < Train::K_METRIC_CAPACITY; ++index)
        SA_EXPECT_OK(metrics.add(Metric{K_NAMES[index], 0.0, false}));

    SA_EXPECT_FAILURE(metrics.add(Metric{K_NAMES[Train::K_METRIC_CAPACITY], 0.0, false}),
                      StatusCode::InvalidDescription, "at most 8", "'i'");
    EXPECT_EQ(metrics.all().size(), Train::K_METRIC_CAPACITY);
}

/** @brief Tests that a fraction prints as a percentage and any other number as it is. */
TEST(Unit_Metrics, AFractionPrintsAsAPercentage)
{
    EXPECT_EQ(Train::describe_metric_value(Metric{"accuracy", 0.9240, true}), "92.40%");
    EXPECT_EQ(Train::describe_metric_value(Metric{"perplexity", 12.5, false}), "12.5");
}

/** @brief Tests that only_output returns the tensor of a step that published one output. */
TEST(Unit_Metrics, OnlyOutputReturnsTheOneTensor)
{
    const SushiBLAS::Tensor tensor{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> one{Train::StepOutput{"output", &tensor}};

    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const found,
                            Train::only_output(view_of(batch, one)));
    EXPECT_EQ(found, &tensor);
}

/** @brief Tests that only_output refuses no output, two outputs and an output with no tensor. */
TEST(Unit_Metrics, OnlyOutputRefusesAnyOtherCount)
{
    const SushiBLAS::Tensor tensor{};
    const Data::Batch batch{};
    const std::vector<Train::StepOutput> two{Train::StepOutput{"p3", &tensor},
                                             Train::StepOutput{"p4", &tensor}};
    const std::vector<Train::StepOutput> unbound{Train::StepOutput{"output", nullptr}};

    SA_EXPECT_FAILURE(Train::only_output(view_of(batch, {})), StatusCode::InvalidDescription,
                      "published 0");
    SA_EXPECT_FAILURE(Train::only_output(view_of(batch, two)), StatusCode::InvalidDescription,
                      "published 2");
    SA_EXPECT_FAILURE(Train::only_output(view_of(batch, unbound)),
                      StatusCode::InvalidDescription, "'output'", "no tensor");
}
