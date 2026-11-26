/**************************************************************************/
/* test_row_feed.cpp                                                      */
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
 * @file test_row_feed.cpp
 * @brief Tests that a row feed writes the rows of a batch into a setup's inputs.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiAI/allocation/host_copy.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include "registries.hpp"
#include "row_feed.hpp"
#include "tiny_run_files.hpp"
#include "training_setup.hpp"

using namespace SushiAI;
namespace TinyRun = SushiAI::Apps::TinyRunTest;

namespace
{
    /** @brief The rows of each batch the setup is traced over and the feed reads. */
    constexpr std::int64_t K_ROWS = 2;

    /** @brief Gives each case a setup of the tiny run traced over K_ROWS rows. */
    class Medium_RowFeed : public Test::EngineTest
    {
        protected:
            /** @brief Builds the setup on the shared backend and enters epoch 0 of its data. */
            void SetUp() override
            {
                spec_ = TinyRun::tiny_spec("");
                const Config::ModelSpec model =
                    SushiAI::Test::require_value(Config::load_model(spec_.model_path));
                setup_ = SushiAI::Test::require_value(Apps::TrainingSetup::make(
                    Apps::compose_registries().value(), spec_, model, backend(), 1, K_ROWS));
                Data::begin_epoch(setup_->dataset(), 0);
            }

            /** @brief The training file the setup was built from; outlives the setup. */
            Config::TrainSpec spec_;

            /** @brief The data, the training graph and the pool the feed writes into. */
            std::unique_ptr<Apps::TrainingSetup> setup_;
    };
} // namespace

/** @brief Tests that a feed writes the rows it was given into the input leaf, row for row. */
TEST_F(Medium_RowFeed, AFeedCopiesItsRowsIntoTheInputs)
{
    const Data::RowRange last_rows{TinyRun::K_BATCH_ROWS - K_ROWS, K_ROWS};
    Data::Batch expected;
    SA_EXPECT_OK(setup_->dataset().read_rows(0, last_rows, expected));
    Apps::RowFeed feed(backend(), *setup_);

    SA_EXPECT_OK(feed.feed(0, last_rows));

    const SushiBLAS::Tensor& inputs = setup_->pool().tensor(setup_->built().spec.inputs);
    EXPECT_EQ(SushiAI::Test::require_value(read_reals(inputs)),
              SushiAI::Test::require_value(read_reals(expected.inputs)));
}

/** @brief Tests that a feed of rows that run past the end of the batch is refused. */
TEST_F(Medium_RowFeed, AFeedPastTheBatchIsRefused)
{
    Apps::RowFeed feed(backend(), *setup_);

    SA_EXPECT_FAILURE(feed.feed(0, Data::RowRange{TinyRun::K_BATCH_ROWS - 1, K_ROWS}),
                      StatusCode::InvalidDescription, "Rows 7 to 9");
}
