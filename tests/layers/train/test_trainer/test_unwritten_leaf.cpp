/**************************************************************************/
/* test_unwritten_leaf.cpp                                                */
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
 * @file test_unwritten_leaf.cpp
 * @brief Tests that Trainer::run_step refuses an objective whose feed leaves a leaf unwritten.
 * @author Mustafa Garip
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/input.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/in_memory_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_output.hpp>
#include <SushiAI/train/row_mask.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_graph.hpp>

using namespace SushiAI;

// SaTest keeps SushiAI::Test reachable inside TEST_F; see tests/layers/train/README.md.
namespace SaTest = SushiAI::Test;

namespace
{
    /** @brief Holds the samples of the batch. */
    constexpr std::int64_t K_BATCH = 4;

    /** @brief Holds the features of a sample. */
    constexpr std::int64_t K_FEATURES = 3;

    /** @brief Holds the classes the objective scores. */
    constexpr std::int64_t K_CLASSES = 3;

    /**
     * @brief Scripts @p objective to declare the leaves of a cross-entropy objective with a row
     *        mask and to feed only the targets, so the weights leaf of the mask stays as it was.
     *
     * The loss is the one the cross-entropy objective emits over the same selection.
     */
    void script_skipping(Train::InMemoryObjective& objective, Train::ObjectiveSelection selection)
    {
        const Train::CrossEntropyObjective inner(K_BATCH, K_CLASSES, DType::Float32,
                                                 std::move(selection));
        objective.script_input_leaf("targets", Shape{K_BATCH, K_CLASSES});
        objective.script_input_leaf(std::string(Train::K_TYPE_MASK_WEIGHTS_LEAF),
                                    Shape{K_BATCH, K_CLASSES});
        objective.script_loss_emitter(
            [inner](const Graph::GraphBuilder& builder, const Graph::Graph& graph,
                    const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves& leaves)
                -> Result<Graph::ValueID>
            {
                SA_TRY_ASSIGN(const Train::LossTrace loss_trace,
                              inner.loss(builder, graph, outputs, leaves));
                const Graph::ValueID loss = loss_trace.loss();
                return loss;
            });
        objective.script_feed_fill("targets", 0.0);
        objective.script_fed_by_batch(true);
    }

    /** @brief Provides the engine the step is compiled on. */
    class Medium_TrainerUnwrittenLeaf : public SaTest::EngineTest
    {

    };
} // namespace

/**
 * @brief Tests that a step is refused, naming the leaf, when the feed skipped one.
 *
 * The objective declares "targets" and "type_mask_weights" and its scripted feed writes the
 * first alone.
 */
TEST_F(Medium_TrainerUnwrittenLeaf, RunStepRefusesALeafTheFeedSkipped)
{
    SA_ASSERT_OK_AND_ASSIGN(const Train::RowMask mask, Train::RowMask::make(0, 0));
    Train::ObjectiveSelection selection;
    selection.mask = mask;
    Train::InMemoryObjective objective;
    script_skipping(objective, std::move(selection));

    NN::Linear model{K_FEATURES, K_CLASSES};
    SA_ASSERT_OK_AND_ASSIGN(
        Train::TrainingGraph built,
        Train::build_training_graph(model, objective, Shape{K_BATCH, K_FEATURES}));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState state, NN::ModelState::create(*engine, model));
    Graph::TensorPool pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(std::vector<Optim::ParameterSlot> slots,
                            Train::bind_parameters(backend(), built, state, pool));
    Optim::SGDOptions sgd;
    sgd.learning_rate = 0.0;
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Optim::SGD> optimizer,
                            Optim::SGD::make(backend(), slots, sgd));
    SA_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<Train::Trainer> trainer,
        Train::Trainer::make(backend(), built, pool, objective, *optimizer));
    const std::vector<Data::TargetDescription> roles = {
        Data::TargetDescription{std::string(Data::K_ROLE_LABELS), Shape{}, DType::Int32,
                                K_CLASSES}};
    SA_ASSERT_OK_AND_ASSIGN(
        Data::Batch batch,
        Data::allocate_batch(*engine, K_BATCH,
                             Data::InputDescription{Shape{K_FEATURES}, DType::Float32}, roles));
    SaTest::write_tensor(batch.inputs, std::vector<double>(K_BATCH * K_FEATURES, 1.0));
    SaTest::write_tensor(batch.targets.front().values, std::vector<double>{0.0, 1.0, 2.0, 1.0});

    SA_EXPECT_FAILURE(trainer->run_step(batch), StatusCode::InvalidDescription,
                      "'type_mask_weights'", "unwritten");
}
