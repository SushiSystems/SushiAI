/**************************************************************************/
/* decision_session.cpp                                                   */
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
 * @file decision_session.cpp
 * @brief Implements the training and the reading of a decision model.
 * @author Mustafa Garip
 */

#include <SushiAI/language/decision_models/decision_session.hpp>

#include <cstddef>
#include <string_view>
#include <utility>

#include <SushiAI/data/target.hpp>
#include <SushiAI/language/blocks/decision_heads.hpp>
#include <SushiAI/language/datasets/decision_dataset.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/host_values.hpp>

#include "decision_run.hpp"

namespace SushiAI::Language::DecisionModels
{
    namespace
    {
        /** @brief Appends @p count reals of @p tensor to @p out. */
        [[nodiscard]] Status append_reals(const SushiBLAS::Tensor& tensor, std::int64_t count,
                                          std::string_view what, std::vector<float>& out) noexcept
        {
            SA_TRY(Train::require_host_reals(tensor, count, what));

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                for (std::int64_t index = 0; index < count; ++index)
                    out.push_back(static_cast<float>(Train::host_real(tensor, index)));

                return Status();
            });
        }

        /** @brief Returns the type of the row whose type mask starts at @p mask. */
        [[nodiscard]] Datasets::DecisionType type_of_row(const float* mask) noexcept
        {
            std::int64_t best = 0;
            for (std::int64_t column = 1; column < Datasets::K_DECISION_TYPES; ++column)
            {
                if (mask[column] > mask[best])
                    best = column;
            }

            return static_cast<Datasets::DecisionType>(best);
        }

        /** @brief Appends the labels, the values and the types of @p batch to @p out. */
        [[nodiscard]] Status append_answers(const Data::Batch& batch, std::int64_t rows,
                                            DecisionPredictions& out) noexcept
        {
            SA_TRY_ASSIGN(const Data::BatchTarget* const choice,
                          Train::require_batch_role(batch, Datasets::K_ROLE_CHOICE));
            SA_TRY_ASSIGN(const Data::BatchTarget* const yes_no,
                          Train::require_batch_role(batch, Datasets::K_ROLE_YES_NO));
            SA_TRY_ASSIGN(const Data::BatchTarget* const number,
                          Train::require_batch_role(batch, Datasets::K_ROLE_NUMBER));
            SA_TRY_ASSIGN(const Data::BatchTarget* const mask,
                          Train::require_batch_role(batch, Datasets::K_ROLE_TYPE_MASK));
            SA_FAIL_IF(choice->values.dtype != DType::Int32
                           || choice->values.num_elements != rows,
                       StatusCode::Incompatible,
                       "the batch holds {} choices of {}; the session reads {} of int32",
                       choice->values.num_elements, dtype_name(choice->values.dtype), rows);
            SA_TRY(Train::require_host_reals(yes_no->values, rows, "the yes-or-no answers"));
            SA_TRY(Train::require_host_reals(number->values, rows, "the numbers"));
            SA_TRY(Train::require_host_reals(mask->values, rows * Datasets::K_DECISION_TYPES,
                                             "the type mask"));

            return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
            {
                const std::int32_t* const index = choice->values.data_as<std::int32_t>();
                for (std::int64_t row = 0; row < rows; ++row)
                {
                    float kinds[Datasets::K_DECISION_TYPES];
                    for (std::int64_t column = 0; column < Datasets::K_DECISION_TYPES; ++column)
                    {
                        kinds[column] = static_cast<float>(Train::host_real(
                            mask->values, row * Datasets::K_DECISION_TYPES + column));
                    }

                    // A row answers with the role of its type; the other roles hold 0 there.
                    const Datasets::DecisionType type = type_of_row(kinds);
                    const Data::BatchTarget* const answers =
                        type == Datasets::DecisionType::Number ? number : yes_no;

                    out.labels.push_back(index[row]);
                    out.values.push_back(static_cast<float>(Train::host_real(answers->values, row)));
                    out.types.push_back(type);
                }

                return Status();
            });
        }

        /** @brief Refuses what a decision session does not run: a schedule, a cluster, a start. */
        [[nodiscard]] Status require_plain_run(const Config::TrainSpec& train) noexcept
        {
            SA_FAIL_IF(!train.schedule.is_null() || !train.cluster.is_null(),
                       StatusCode::Unsupported,
                       "a decision session runs the epochs of a training file with its optimizer "
                       "as it states it; the file states a 'schedule' or a 'cluster'");
            SA_FAIL_IF(!train.weights.empty() || !train.resume.empty(), StatusCode::Unsupported,
                       "a decision session starts from the initialised model; the file states "
                       "'weights' or 'resume'");

            return Status();
        }
    } // namespace

    DecisionSession::DecisionSession(NN::ModelGraph model, std::int64_t epochs) noexcept
        : epochs_(epochs), model_(std::move(model))
    {

    }

    DecisionSession::~DecisionSession() = default;

    Result<std::unique_ptr<DecisionSession>> DecisionSession::make(
        ExecutionBackend& backend, const DecisionSessionOptions& options,
        const Data::Dataset& training) noexcept
    {
        SA_FATAL_IF(options.nodes == nullptr || options.objectives == nullptr,
                    "a decision session was given no node registry or no objective registry");
        SA_TRY(require_plain_run(options.train));

        SA_TRY_ASSIGN(NN::ModelGraph model, NN::build_model(*options.nodes, options.model));
        SA_TRY_ASSIGN(std::unique_ptr<DecisionSession> session,
                      SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<DecisionSession>(
                new DecisionSession(std::move(model), options.train.epochs));
        }));

        SA_TRY_ASSIGN(session->training_,
                      DecisionRun::make(RunKind::Training, backend, session->model_,
                                        session->state_, options, training));
        SA_TRY_ASSIGN(session->probe_,
                      DecisionRun::make(RunKind::Probe, backend, session->model_,
                                        session->state_, options, training));

        return session;
    }

    Result<Train::RunReport> DecisionSession::fit(const Data::Dataset& dataset) noexcept
    {
        training_->trainer().options().epochs = epochs_;

        return training_->trainer().fit(dataset);
    }

    Result<DecisionPredictions> DecisionSession::predict(const Data::Dataset& dataset) noexcept
    {
        SA_FAIL_IF(Data::find_target(dataset.targets(), Datasets::K_ROLE_CHOICE) == nullptr,
                   StatusCode::InvalidDescription,
                   "the dataset '{}' publishes no '{}' role, which a decision session reads",
                   dataset.name(), Datasets::K_ROLE_CHOICE);

        DecisionPredictions predictions;
        const std::int64_t rows = dataset.batch_size();
        for (std::size_t index = 0; index < dataset.batch_count(); ++index)
        {
            Data::Batch batch;
            SA_TRY(dataset.read(index, batch));
            SA_TRY(probe_->trainer().run_step(batch));

            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const choice,
                          probe_->output(Blocks::K_CHOICE_PORT));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const yes_no,
                          probe_->output(Blocks::K_YES_NO_PORT));
            SA_TRY_ASSIGN(const SushiBLAS::Tensor* const number,
                          probe_->output(Blocks::K_NUMBER_PORT));

            // The choice output is [rows, options]; its width is the options of every record.
            predictions.options = choice->num_elements / rows;
            SA_TRY(append_reals(*choice, choice->num_elements, "the choice logits",
                                predictions.choice_logits));
            SA_TRY(append_reals(*yes_no, rows, "the yes-or-no logits", predictions.yes_no_logits));
            SA_TRY(append_reals(*number, rows, "the numbers", predictions.numbers));
            SA_TRY(append_answers(batch, rows, predictions));
        }

        return predictions;
    }
} // namespace SushiAI::Language::DecisionModels
