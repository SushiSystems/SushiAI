/**************************************************************************/
/* test_table_dataset.cpp                                                 */
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
 * @file test_table_dataset.cpp
 * @brief Tests the dataset that collates each batch of a tabular sample list.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/SushiAI.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/table_dataset.hpp>

using namespace SushiAI;
using namespace SushiAI::Vision::Datasets;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides an engine and the builders of the lists these tests read. */
    class Unit_DatasetsTableDataset : public SaTest::EngineTest
    {
        protected:
            /**
             * @brief Returns a list of rows of two features and one target.
             * @param count Rows; row i holds 10 i, 10 i + 1 and 10 i + 2.
             */
            SampleList table_list(std::size_t count) const
            {
                SampleList list;
                list.files.push_back("rows.csv");
                list.table.feature_count = 2;
                list.table.target_count = 1;
                list.samples.resize(count);

                for (std::size_t i = 0; i < count * 3; ++i)
                    list.table.values.push_back(static_cast<float>(10 * (i / 3) + i % 3));

                return list;
            }

            /** @brief Returns the options of batches of two rows. */
            TableDatasetOptions options() const
            {
                TableDatasetOptions made;
                made.batch_size = 2;
                made.name = "rows";
                return made;
            }

            /** @brief Returns a regression collator of two features and one target. */
            std::unique_ptr<const Collator> regression()
            {
                Result<std::unique_ptr<RegressionCollator>> made = RegressionCollator::make(2, 1);
                EXPECT_TRUE(made.status().is_ok());
                return std::move(made).value();
            }
    };
} // namespace

/** @brief Tests that the dataset reports the batch size, the shapes and the whole batches. */
TEST_F(Unit_DatasetsTableDataset, ItDescribesItsBatches)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TableDataset> dataset,
        TableDataset::make(*engine, table_list(5), regression(), options()));

    EXPECT_EQ(dataset->name(), "rows");
    EXPECT_EQ(dataset->batch_size(), 2);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->sample_shape(), Shape{2});
    ASSERT_EQ(dataset->targets().size(), 1u);
    EXPECT_EQ(dataset->targets()[0].role, Data::K_ROLE_VALUES);
    EXPECT_EQ(dataset->targets()[0].sample_shape, Shape{1});
}

/** @brief Tests that a batch holds its rows in list order, read through Data::Dataset. */
TEST_F(Unit_DatasetsTableDataset, ABatchHoldsItsRowsInListOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TableDataset> made,
        TableDataset::make(*engine, table_list(5), regression(), options()));
    const Data::Dataset& dataset = *made;

    Data::Batch batch;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(1, batch)));

    // The values are copied and never computed, so they compare exactly.
    const float* inputs = batch.inputs.data_as<float>();
    const float* values = batch.targets[0].values.data_as<float>();
    EXPECT_EQ(std::vector<float>(inputs, inputs + 4),
              (std::vector<float>{20.0f, 21.0f, 30.0f, 31.0f}));
    EXPECT_EQ(std::vector<float>(values, values + 2), (std::vector<float>{22.0f, 32.0f}));

    const float* const first = inputs;
    ASSERT_TRUE(SaTest::status_is_ok(dataset.read(0, batch)));
    EXPECT_EQ(batch.inputs.data_as<float>(), first);
    EXPECT_EQ(first[0], 0.0f);
}

/** @brief Tests that make refuses a null collator, a list with no table and bad counts. */
TEST_F(Unit_DatasetsTableDataset, MakeRefusesWhatItCannotBatch)
{
    SA_EXPECT_FAILURE(TableDataset::make(*engine, table_list(4), nullptr, options()),
                      StatusCode::InvalidDescription, "TableDataset", "'rows' has no collator");

    SampleList images;
    images.samples.resize(4);
    SA_EXPECT_FAILURE(TableDataset::make(*engine, images, regression(), options()),
                      StatusCode::InvalidDescription, "a table of 0 features");

    SampleList ragged = table_list(4);
    ragged.table.values.pop_back();
    SA_EXPECT_FAILURE(TableDataset::make(*engine, ragged, regression(), options()),
                      StatusCode::InvalidDescription, "4 samples and 11 numbers",
                      "a row holds 3");

    TableDatasetOptions empty = options();
    empty.batch_size = 0;
    SA_EXPECT_FAILURE(TableDataset::make(*engine, table_list(4), regression(), empty),
                      StatusCode::InvalidDescription, "a batch size of 0");

    SA_EXPECT_FAILURE(TableDataset::make(*engine, table_list(1), regression(), options()),
                      StatusCode::InvalidDescription, "1 samples", "one batch of 2");
}

/** @brief Tests that a batch index past the last batch stops the process. */
TEST_F(Unit_DatasetsTableDataset, ABatchOutsideTheEpochStopsTheProcess)
{
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TableDataset> dataset,
        TableDataset::make(*engine, table_list(2), regression(), options()));

    Data::Batch batch;
    SA_EXPECT_FATAL((void)dataset->read(1, batch), "TableDataset", "1 batches", "batch 1");
}

/** @brief Tests that the failure of the collator reaches the caller of read. */
TEST_F(Unit_DatasetsTableDataset, TheFailureOfTheCollatorIsReturned)
{
    Result<std::unique_ptr<RegressionCollator>> wide = RegressionCollator::make(2, 2);
    ASSERT_TRUE(wide.status().is_ok());

    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<TableDataset> dataset,
        TableDataset::make(*engine, table_list(2), std::move(wide).value(), options()));

    Data::Batch batch;
    SA_EXPECT_FAILURE(dataset->read(0, batch), StatusCode::InvalidDescription, "rows.csv",
                      "RegressionCollator", "2 features and 1 targets");
}
