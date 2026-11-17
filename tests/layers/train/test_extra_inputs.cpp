/**************************************************************************/
/* test_extra_inputs.cpp                                                  */
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
 * @file test_extra_inputs.cpp
 * @brief Tests the extra inputs of a model through the step builders, the trainer and the
 *        evaluation step: the leaf of each, the role that feeds it and the copy of a batch.
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
#include <SushiRuntime/core/span.hpp>
#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/roles.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/engine_fixture/engine_test.hpp>
#include <SushiAI/execution/execution.hpp>
#include <SushiAI/execution/host_execution_backend.hpp>
#include <SushiAI/failure/result.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/failure_expectations/status_expectations.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/input_leaves.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/sample_models/extra_input_model.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/tensor_values/tensor_values.hpp>
#include <SushiAI/train/evaluation_step.hpp>
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
    /** @brief Counts the samples of every batch of this file. */
    constexpr std::int64_t K_BATCH = SaTest::K_EXTRA_BATCH;

    /** @brief Counts the elements of one sample of the input and of the extra input. */
    constexpr std::int64_t K_ELEMENTS = SaTest::K_EXTRA_MODEL_ELEMENTS;

    /** @brief Counts the classes the model ends in. */
    constexpr std::int64_t K_CLASSES = SaTest::K_EXTRA_MODEL_CLASSES;

    /** @brief Holds the leaf and the role the extra input of the sample model has. */
    constexpr std::string_view K_OFFSET = SaTest::K_EXTRA_INPUT_OFFSET;

    using SaTest::eighths;

    /** @brief Returns the extents of the input of one batch: [K_BATCH, 2, 2, 3]. */
    Shape batch_extents()
    {
        return SaTest::extra_batch_extents();
    }

    /** @brief Returns the elementwise sum of @p left and @p right, which have as many values. */
    std::vector<double> sum_of(const std::vector<double>& left, const std::vector<double>& right)
    {
        std::vector<double> sums;
        for (std::size_t index = 0; index < left.size(); ++index)
            sums.push_back(left[index] + right[index]);

        return sums;
    }

    /** @brief Expects @p step to hold one leaf for the offset, of the shape of a batch. */
    void expect_declares_the_offset(const Train::TracedStep& step)
    {
        ASSERT_EQ(step.extra_inputs.leaves().size(), 1u);
        EXPECT_EQ(step.extra_inputs.leaves()[0].name, K_OFFSET);
        EXPECT_EQ(step.graph.value(step.extra_inputs.leaves()[0].value).shape, batch_extents());
    }

    /** @brief Returns a batch of @p inputs, the labels 0 1 2 1 and the role "offset" of @p offsets. */
    Result<Data::Batch> make_batch(SushiBLAS::Engine& engine, const std::vector<double>& inputs,
                                   const std::vector<double>& offsets,
                                   const Shape& offset_extents = batch_extents())
    {
        return SaTest::make_extra_input_batch(engine, inputs, offsets, offset_extents);
    }

    /** @brief Holds the leaves of a model of two extra inputs and a batch that holds both roles. */
    class Small_ExtraFeeds : public ::testing::Test
    {
        protected:
            /** @brief Returns a tensor of @p count float32 values. */
            [[nodiscard]] Result<SushiBLAS::Tensor> tensor_of(std::int64_t count)
            {
                SA_TRY_ASSIGN(SushiBLAS::Tensor tensor,
                              allocate_tensor(backend.engine(), Shape{count}, DType::Float32));
                SaTest::write_tensor(tensor, eighths(count, 1));
                return tensor;
            }

            /** @brief Returns a batch with the roles "scale", "offset" and "labels", in that order. */
            [[nodiscard]] Result<Data::Batch> batch_of_roles()
            {
                Data::Batch batch;
                for (const char* role : {"scale", "offset", "labels"})
                {
                    Data::BatchTarget target;
                    target.role = role;
                    SA_TRY_ASSIGN(target.values, tensor_of(3));
                    batch.targets.push_back(std::move(target));
                }

                return batch;
            }

            /** @brief The host backend whose engine allocates the tensors. */
            HostExecutionBackend backend;
    };

    /** @brief Provides the engine, a trainer over the sample model and an evaluation step. */
    class Medium_ExtraInputsRun : public SaTest::EngineTest
    {
        protected:
            /** @brief Traces the sample model, binds it and builds an SGD trainer of @p rate. */
            void prepare(double rate)
            {
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph built_model,
                                        SaTest::build_extra_input_model());
                model = std::make_unique<NN::ModelGraph>(std::move(built_model));
                objective = std::make_unique<Train::CrossEntropyObjective>(K_BATCH, K_CLASSES);
                SA_ASSERT_OK_AND_ASSIGN(built,
                                        Train::build_training_graph(*model, *objective,
                                                                    batch_extents()));
                SA_ASSERT_OK_AND_ASSIGN(NN::ModelState created,
                                        NN::ModelState::create(*engine, *model));
                state = std::make_unique<NN::ModelState>(std::move(created));
                pool = std::make_unique<Graph::TensorPool>(*engine);
                SA_ASSERT_OK_AND_ASSIGN(slots,
                                        Train::bind_parameters(backend(), built, *state, *pool));
                // The head is the only layer, so a zeroed final layer would make every logit 0.
                NN::InitPolicy policy;
                policy.zero_final_layer = false;
                SA_EXPECT_OK(NN::initialize(backend(), *state, policy));

                Optim::SGDOptions sgd;
                sgd.learning_rate = rate;
                SA_ASSERT_OK_AND_ASSIGN(optimizer, Optim::SGD::make(backend(), slots, sgd));
                SA_ASSERT_OK_AND_ASSIGN(trainer,
                                        Train::Trainer::make(backend(), built, *pool, *objective,
                                                             *optimizer));
            }

            /** @brief Runs one step over @p inputs and @p offsets and writes its loss to @p loss. */
            void loss_over(const std::vector<double>& inputs, const std::vector<double>& offsets,
                           double& loss)
            {
                SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                                        make_batch(*engine, inputs, offsets));
                SA_ASSERT_OK_AND_ASSIGN(const Train::StepMetrics metrics,
                                        trainer->run_step(batch));
                loss = metrics.loss;
            }

            std::unique_ptr<NN::ModelGraph> model;                 /**< @brief The sample model. */
            std::unique_ptr<Train::CrossEntropyObjective> objective; /**< @brief Its objective. */
            Train::TrainingGraph built;                            /**< @brief Its training graph. */
            std::unique_ptr<NN::ModelState> state;                 /**< @brief Its parameters. */
            std::unique_ptr<Graph::TensorPool> pool;               /**< @brief Its step's values. */
            std::vector<Optim::ParameterSlot> slots;               /**< @brief Parameters and gradients. */
            std::unique_ptr<Optim::SGD> optimizer;                 /**< @brief Applies the gradients. */
            std::unique_ptr<Train::Trainer> trainer;               /**< @brief Runs the step. */
    };
} // namespace

/** @brief Tests that the training graph declares a leaf of the batch shape for each extra input. */
TEST(Small_ExtraInputsGraph, TheTrainingGraphDeclaresALeafOfTheBatchShapeForEachExtraInput)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_extra_input_model());
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, batch_extents()));

    ASSERT_EQ(built.extra_inputs.leaves().size(), 1u);
    const NN::InputLeaf& leaf = built.extra_inputs.leaves()[0];
    EXPECT_EQ(leaf.name, K_OFFSET);
    EXPECT_NE(leaf.value, built.spec.inputs);
    EXPECT_EQ(built.graph.value(leaf.value).shape, batch_extents());
    EXPECT_EQ(built.graph.value(leaf.value).dtype, DType::Float32);
    EXPECT_EQ(built.extra_inputs.find(K_OFFSET), leaf.value);
    EXPECT_EQ(built.extra_inputs.find("absent"), Graph::K_NO_VALUE);
}

/** @brief Tests that the training graph requires the role of each extra input beside the targets. */
TEST(Small_ExtraInputsGraph, TheTrainingGraphRequiresTheRoleOfEachExtraInput)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_extra_input_model());
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, batch_extents()));

    ASSERT_EQ(built.leaves.required().size(), 2u);
    const Data::TargetDescription* const offset =
        Data::find_target(built.leaves.required(), K_OFFSET);
    ASSERT_NE(offset, nullptr);
    EXPECT_EQ(offset->sample_shape, SaTest::extra_model_sample());
    EXPECT_EQ(offset->dtype, DType::Float32);
    EXPECT_EQ(offset->classes, 0);
    EXPECT_NE(Data::find_target(built.leaves.required(), Data::K_ROLE_LABELS), nullptr);
    EXPECT_FALSE(built.leaves.has(K_OFFSET));
}

/** @brief Tests that the probe, the evaluation and the carrying graphs declare the leaf too. */
TEST(Small_ExtraInputsGraph, TheProbeEvaluationAndCarryingGraphsDeclareTheLeafToo)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph model, SaTest::build_extra_input_model());

    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep probe,
                            Train::build_probe_graph(model, batch_extents()));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep evaluation,
                            Train::build_evaluation_graph(model, batch_extents()));
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep carrying,
                            Train::build_carrying_graph(model, batch_extents()));

    expect_declares_the_offset(probe);
    expect_declares_the_offset(evaluation);
    expect_declares_the_offset(carrying);
}

/** @brief Tests that a model with no extra input declares none and requires no further role. */
TEST(Small_ExtraInputsGraph, AModelWithNoExtraInputDeclaresNone)
{
    NN::Sequential<NN::Linear, NN::ReLU, NN::Linear> model(NN::Linear{6, 8}, NN::ReLU{},
                                                           NN::Linear{8, K_CLASSES});
    const Train::CrossEntropyObjective objective(K_BATCH, K_CLASSES);

    SA_ASSERT_OK_AND_ASSIGN(const Train::TrainingGraph built,
                            Train::build_training_graph(model, objective, Shape{K_BATCH, 6}));

    EXPECT_TRUE(built.extra_inputs.leaves().empty());
    EXPECT_EQ(built.leaves.required().size(), 1u);
}

/** @brief Tests that an extra input named as the model input of every step is refused. */
TEST(Small_ExtraInputsGraph, AnExtraInputNamedAsTheModelInputIsRefused)
{
    NN::ModelGraph model(Shape{6});
    SA_ASSERT_OK_AND_ASSIGN(const NN::NodeRef clash,
                            model.declare_input(std::string(Train::K_MODEL_INPUT_NAME), Shape{6},
                                                DType::Float32));
    SA_EXPECT_OK(model.output("echo", clash));

    SA_EXPECT_FAILURE(Train::build_probe_graph(model, Shape{K_BATCH, 6}),
                      StatusCode::InvalidDescription, "'inputs'", "model input");
}

/** @brief Tests that extra_feeds pairs each leaf with the batch role of its name, in leaf order. */
TEST_F(Small_ExtraFeeds, EachLeafIsPairedWithTheRoleOfItsNameInLeafOrder)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of_roles());
    NN::InputLeafList leaves;
    leaves.add(NN::InputLeaf{"offset", 5});
    leaves.add(NN::InputLeaf{"scale", 6});
    std::vector<Train::NamedFeed> feeds;

    SA_EXPECT_OK(Train::extra_feeds(leaves.leaves(), batch, feeds));

    ASSERT_EQ(feeds.size(), 2u);
    EXPECT_EQ(feeds[0].name, "offset");
    EXPECT_EQ(feeds[0].tensor, &batch.targets[1].values);
    EXPECT_EQ(feeds[1].name, "scale");
    EXPECT_EQ(feeds[1].tensor, &batch.targets[0].values);
}

/** @brief Tests that extra_feeds appends to the feeds it is given and keeps what they held. */
TEST_F(Small_ExtraFeeds, TheFeedsAreAppendedAndWhatTheyHeldIsKept)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of_roles());
    NN::InputLeafList leaves;
    leaves.add(NN::InputLeaf{"offset", 5});
    std::vector<Train::NamedFeed> feeds{Train::NamedFeed{"inputs", &batch.inputs}};

    SA_EXPECT_OK(Train::extra_feeds(leaves.leaves(), batch, feeds));

    ASSERT_EQ(feeds.size(), 2u);
    EXPECT_EQ(feeds[0].name, "inputs");
    EXPECT_EQ(feeds[1].name, "offset");
}

/** @brief Tests that a batch without the role of a leaf is refused naming the leaf. */
TEST_F(Small_ExtraFeeds, ABatchWithoutTheRoleOfALeafIsRefusedNamingTheLeaf)
{
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch, batch_of_roles());
    NN::InputLeafList leaves;
    leaves.add(NN::InputLeaf{"offset", 5});
    leaves.add(NN::InputLeaf{"lengths", 6});
    std::vector<Train::NamedFeed> feeds;

    SA_EXPECT_FAILURE(Train::extra_feeds(leaves.leaves(), batch, feeds),
                      StatusCode::InvalidDescription, "'lengths'", "extra input");
}

/** @brief Tests that a training step copies each extra input from its batch role into its leaf. */
TEST_F(Medium_ExtraInputsRun, ATrainingStepCopiesTheRoleIntoTheLeaf)
{
    prepare(0.0);
    const std::vector<double> offsets = eighths(K_BATCH * K_ELEMENTS, 5);
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            make_batch(*engine, eighths(K_BATCH * K_ELEMENTS, 3), offsets));

    SA_EXPECT_OK(trainer->run_step(batch));

    const Graph::ValueID leaf = built.extra_inputs.find(K_OFFSET);
    ASSERT_NE(leaf, Graph::K_NO_VALUE);
    EXPECT_EQ(SaTest::read_tensor(pool->tensor(leaf)), offsets);
}

/** @brief Tests that the extra input reaches the loss: another offset moves it, the same does not. */
TEST_F(Medium_ExtraInputsRun, TheExtraInputMovesTheLossAndTheSameBatchRepeatsIt)
{
    prepare(0.0);
    const std::vector<double> inputs = eighths(K_BATCH * K_ELEMENTS, 3);
    const std::vector<double> first = eighths(K_BATCH * K_ELEMENTS, 5);
    const std::vector<double> second = eighths(K_BATCH * K_ELEMENTS, 7);

    double loss_first = 0.0;
    double loss_second = 0.0;
    double loss_first_again = 0.0;
    loss_over(inputs, first, loss_first);
    loss_over(inputs, second, loss_second);
    loss_over(inputs, first, loss_first_again);

    // A rate of 0 keeps the weights, so only the offset can move the loss.
    EXPECT_NE(loss_first, loss_second);
    EXPECT_EQ(loss_first, loss_first_again);
}

/** @brief Tests that a batch with no role of the name, or another shape of it, is refused. */
TEST_F(Medium_ExtraInputsRun, AStepRefusesABatchWithoutTheRoleOrWithAnotherShape)
{
    prepare(0.0);
    const std::vector<double> inputs = eighths(K_BATCH * K_ELEMENTS, 3);
    SA_ASSERT_OK_AND_ASSIGN(Data::Batch without,
                            make_batch(*engine, inputs, eighths(K_BATCH * K_ELEMENTS, 5)));
    without.targets.pop_back();
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch narrow,
        make_batch(*engine, inputs, eighths(K_BATCH * 5, 5), Shape{K_BATCH, 5}));

    SA_EXPECT_FAILURE(trainer->run_step(without), StatusCode::InvalidDescription, "'offset'");
    SA_EXPECT_FAILURE(trainer->run_step(narrow), StatusCode::Incompatible, "offset", "[4, 5]",
                      "[4, 2, 2, 3]");
}

/** @brief Tests that an evaluation step fed from a batch reads the extra input as the sum it is. */
TEST_F(Medium_ExtraInputsRun, AnEvaluationStepReadsTheExtraInputFromTheBatch)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph eval_model, SaTest::build_extra_input_model());
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                            Train::build_evaluation_graph(eval_model, batch_extents()));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState eval_state,
                            NN::ModelState::create(*engine, eval_model));
    SA_EXPECT_OK(NN::initialize(backend(), eval_state));
    Graph::TensorPool eval_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend(), traced, eval_state, eval_pool));

    // The model adds the offset to the input first, so an offset moved into the input gives the same sum.
    const std::vector<double> inputs = eighths(K_BATCH * K_ELEMENTS, 3);
    const std::vector<double> offsets = eighths(K_BATCH * K_ELEMENTS, 5);
    const std::vector<double> sums = sum_of(inputs, offsets);

    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch split, make_batch(*engine, inputs, offsets));
    SA_ASSERT_OK_AND_ASSIGN(
        const Data::Batch merged,
        make_batch(*engine, sums, std::vector<double>(sums.size(), 0.0)));

    SA_EXPECT_OK(step->run_batch(split));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const first, step->output("logits"));
    const std::vector<double> logits_of_split = SaTest::read_tensor(*first);
    SA_EXPECT_OK(step->run_batch(merged));
    SA_ASSERT_OK_AND_ASSIGN(const SushiBLAS::Tensor* const second, step->output("logits"));

    EXPECT_EQ(SaTest::read_tensor(*second), logits_of_split);
}

/** @brief Tests that a run by position of a step with an extra input is refused with the leaves. */
TEST_F(Medium_ExtraInputsRun, ARunByPositionOfAStepWithAnExtraInputNamesBothLeaves)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph eval_model, SaTest::build_extra_input_model());
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                            Train::build_evaluation_graph(eval_model, batch_extents()));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState eval_state,
                            NN::ModelState::create(*engine, eval_model));
    SA_EXPECT_OK(NN::initialize(backend(), eval_state));
    Graph::TensorPool eval_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend(), traced, eval_state, eval_pool));
    SA_ASSERT_OK_AND_ASSIGN(const Data::Batch batch,
                            make_batch(*engine, eighths(K_BATCH * K_ELEMENTS, 3),
                                       eighths(K_BATCH * K_ELEMENTS, 5)));

    ASSERT_EQ(step->input_names().size(), 2u);
    EXPECT_EQ(step->input_names()[0], "inputs");
    EXPECT_EQ(step->input_names()[1], "offset");
    SA_EXPECT_FAILURE(step->run(batch.inputs), StatusCode::Incompatible, "fed 2", "'offset'");
}

/** @brief Tests that an evaluation step run over a batch without an extra input names that input. */
TEST_F(Medium_ExtraInputsRun, AnEvaluationStepRefusesABatchWithoutTheExtraInputNamingIt)
{
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelGraph eval_model, SaTest::build_extra_input_model());
    SA_ASSERT_OK_AND_ASSIGN(const Train::TracedStep traced,
                            Train::build_evaluation_graph(eval_model, batch_extents()));
    SA_ASSERT_OK_AND_ASSIGN(NN::ModelState eval_state,
                            NN::ModelState::create(*engine, eval_model));
    SA_EXPECT_OK(NN::initialize(backend(), eval_state));
    Graph::TensorPool eval_pool(*engine);
    SA_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<Train::EvaluationStep> step,
                            Train::EvaluationStep::make(backend(), traced, eval_state, eval_pool));
    SA_ASSERT_OK_AND_ASSIGN(Data::Batch without,
                            make_batch(*engine, eighths(K_BATCH * K_ELEMENTS, 3),
                                       eighths(K_BATCH * K_ELEMENTS, 5)));
    without.targets.pop_back();

    SA_EXPECT_FAILURE(step->run_batch(without), StatusCode::InvalidDescription, "'offset'");
}
