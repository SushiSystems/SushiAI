/**************************************************************************/
/* test_extra_input_copy.cpp                                              */
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
 * @file test_extra_input_copy.cpp
 * @brief Tests that copy_extra_inputs copies the role of each extra input into its leaf and refuses
 *        a batch that cannot feed one.
 * @author Mustafa Garip
 */

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/blas/blas.hpp>
#include <SushiBLAS/engine/task_recorder.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/sample_models/extra_input_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/extra_input_copy.hpp>
#include <SushiAI/train/extra_inputs.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <support/extra_input_batches.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Provides the engine, the training graph of the sample model and its pool. */
    class Medium_ExtraInputCopy : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces the sample model and materialises the leaves of its training graph. */
            void prepare()
            {
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built_model,
                                        SaTest::build_extra_input_model());
                model = std::make_unique<NN::ModelGraph>(std::move(built_model));
                objective = std::make_unique<Train::CrossEntropyObjective>(
                    SaTest::K_EXTRA_BATCH, SaTest::K_EXTRA_MODEL_CLASSES);
                SA_ASSERT_OK_AND_ASSIGN(built,
                                        Train::build_training_graph(*model, *objective,
                                                                    SaTest::extra_batch_extents()));
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));
            }

            /** @brief Records and runs copy_extra_inputs of @p batch over the leaves of `built`. */
            [[nodiscard]] Status copy_extras_of(const Data::Batch& batch)
            {
                std::vector<Train::NamedFeed> feeds;
                return SushiAI::Detail::record_and_run(
                    backend(), [&](SushiBLAS::TaskRecorder& recorder) -> Status
                {
                    SushiBLAS::BLASOps blas(recorder);
                    return Train::copy_extra_inputs(built.extra_inputs.leaves(), batch, *pool,
                                                    blas, feeds);
                });
            }

            std::unique_ptr<NN::ModelGraph> model;                   /**< @brief The sample model. */
            std::unique_ptr<Train::CrossEntropyObjective> objective; /**< @brief Its objective. */
            Train::TrainingGraph built;                              /**< @brief Its training graph. */
            std::unique_ptr<NN::ModelState> state;                   /**< @brief Its parameters. */
            std::unique_ptr<Graph::TensorPool> pool;                 /**< @brief Its step's values. */
            std::vector<Optim::ParameterSlot> slots;                 /**< @brief Parameters and gradients. */
    };
} // namespace

/** @brief Tests that copy_extra_inputs copies the role of each extra input into the pool's leaf. */
TEST_F(Medium_ExtraInputCopy, CopyExtraInputsCopiesTheRoleOfEachLeafIntoItsLeaf)
{
    prepare();
    const std::int64_t elements = SaTest::K_EXTRA_BATCH * SaTest::K_EXTRA_MODEL_ELEMENTS;
    const std::vector<double> offsets = SaTest::eighths(elements, 5);
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            SaTest::make_extra_input_batch(*engine, SaTest::eighths(elements, 3),
                                                           offsets));

    SA_EXPECT_OK(copy_extras_of(batch));

    const Graph::ValueID leaf = built.extra_inputs.find(SaTest::K_EXTRA_INPUT_OFFSET);
    ASSERT_NE(leaf, Graph::K_NO_VALUE);
    EXPECT_EQ(SaTest::read_tensor(pool->tensor(leaf)), offsets);
}

/** @brief Tests that copy_extra_inputs refuses a missing role and a role of another shape. */
TEST_F(Medium_ExtraInputCopy, CopyExtraInputsRefusesAMissingRoleAndAnotherShape)
{
    prepare();
    const std::int64_t elements = SaTest::K_EXTRA_BATCH * SaTest::K_EXTRA_MODEL_ELEMENTS;
    const std::vector<double> inputs = SaTest::eighths(elements, 3);
    SA_ASSERT_OK_AND_ASSIGN(Data::Batch without,
                            SaTest::make_extra_input_batch(*engine, inputs,
                                                           SaTest::eighths(elements, 5)));
    without.targets.pop_back();
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch narrow,
        SaTest::make_extra_input_batch(*engine, inputs,
                                       SaTest::eighths(SaTest::K_EXTRA_BATCH * 5, 5),
                                       Shape{SaTest::K_EXTRA_BATCH, 5}));

    SA_EXPECT_FAILURE(copy_extras_of(without), StatusCode::InvalidDescription, "'offset'");
    SA_EXPECT_FAILURE(copy_extras_of(narrow), StatusCode::Incompatible, "offset", "[4, 5]",
                      "[4, 2, 2, 3]");
}
