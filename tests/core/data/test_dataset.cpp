/**************************************************************************/
/* test_dataset.cpp                                                       */
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
 * @file test_dataset.cpp
 * @brief Tests what the generated datasets state, how they read a batch, and what they refuse.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>

using namespace SushiAI;
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Returns the options of a small synthetic set: 2 batches of 4 samples, 3 features. */
    Data::SyntheticOptions small_synthetic()
    {
        Data::SyntheticOptions options;
        options.batch_size = 4;
        options.batches = 2;
        options.features = 3;
        options.classes = 2;
        return options;
    }

    /** @brief Provides the engine the datasets are generated on. */
    class Medium_Dataset : public SaTest::EngineTest
    {
        protected:
            /** @brief Returns the inputs of batch @p index of a synthetic set seeded @p seed. */
            std::vector<float> synthetic_inputs(std::uint64_t seed, std::size_t index)
            {
                Data::SyntheticOptions options = small_synthetic();
                options.seed = seed;
                std::vector<float> values;
                Result<std::unique_ptr<Data::SyntheticClassification>> made =
                    Data::SyntheticClassification::make(*engine, options);
                EXPECT_TRUE(SaTest::status_is_ok(Detail::status_of(made)));
                if (!made.has_value())
                    return values;

                Data::Batch batch;
                EXPECT_TRUE(SaTest::status_is_ok(made.value()->read(index, batch)));
                const float* const data = batch.inputs.data_as<float>();
                values.assign(data, data + batch.inputs.num_elements);
                return values;
            }
    };
} // namespace

/** @brief Tests that a synthetic set states the extents it was made with and its name. */
TEST_F(Medium_Dataset, SyntheticClassificationStatesItsExtents)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, small_synthetic()));

    EXPECT_EQ(dataset->batch_size(), 4);
    EXPECT_EQ(dataset->batch_count(), 2u);
    EXPECT_EQ(dataset->name(), "synthetic-classification");

    Data::Batch batch;
    SA_EXPECT_OK(dataset->read(1, batch));
    ASSERT_EQ(batch.inputs.rank, 2);
    EXPECT_EQ(batch.inputs.shape[0], 4);
    EXPECT_EQ(batch.inputs.shape[1], 3);
}

/** @brief Tests that two reads of one batch share its tensor and copy nothing. */
TEST_F(Medium_Dataset, AReadSharesTheHeldBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Data::SyntheticClassification> dataset,
                            Data::SyntheticClassification::make(*engine, small_synthetic()));
    Data::Batch first;
    Data::Batch second;

    SA_EXPECT_OK(dataset->read(0, first));
    SA_EXPECT_OK(dataset->read(0, second));

    EXPECT_EQ(first.inputs.data(), second.inputs.data());
}

/** @brief Tests that one seed gives the same inputs and another seed other inputs. */
TEST_F(Medium_Dataset, TheSeedFixesTheSyntheticInputs)
{
    const std::vector<float> first = synthetic_inputs(5, 1);

    EXPECT_EQ(first, synthetic_inputs(5, 1));
    EXPECT_NE(first, synthetic_inputs(6, 1));
}

/** @brief Tests that a synthetic set with an extent that is not positive is refused. */
TEST_F(Medium_Dataset, SyntheticClassificationRefusesANonPositiveExtent)
{
    Data::SyntheticOptions options = small_synthetic();
    options.features = 0;

    SA_EXPECT_FAILURE(Data::SyntheticClassification::make(*engine, options),
                      StatusCode::InvalidDescription, "positive extents", "features=0");
}

/** @brief Tests that Rings refuses a negative distractor count, a zero width and negative noise. */
TEST_F(Medium_Dataset, RingsRefusesAnOptionOutOfRange)
{
    Data::RingsOptions distractors;
    distractors.distractor_features = -1;
    Data::RingsOptions width;
    width.ring_width = 0.0;
    Data::RingsOptions noise;
    noise.position_noise = -0.5;

    SA_EXPECT_FAILURE(Data::Rings::make(*engine, distractors), StatusCode::InvalidDescription,
                      "distractor");
    SA_EXPECT_FAILURE(Data::Rings::make(*engine, width), StatusCode::InvalidDescription,
                      "ring width");
    SA_EXPECT_FAILURE(Data::Rings::make(*engine, noise), StatusCode::InvalidDescription,
                      "position noise");
}
