/**************************************************************************/
/* expert_balance_objective.cpp                                           */
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
 * @file expert_balance_objective.cpp
 * @brief Implements the balance objective of the experts, its metrics and its factory.
 * @author Mustafa Garip
 */

#include <SushiAI/language/training/expert_balance_objective.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/seam/seam.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/train/host_values.hpp>

namespace SushiAI::Language::Training
{
    namespace
    {
        /** @brief Reports whether @p name begins with @p prefix. */
        [[nodiscard]] bool begins_with(std::string_view name, std::string_view prefix) noexcept
        {
            return name.substr(0, prefix.size()) == prefix;
        }

        /** @brief Returns the tensor @p step published as @p name, or null for none. */
        [[nodiscard]] const SushiBLAS::Tensor* find_step_output(const Train::StepView& step,
                                                                std::string_view name) noexcept
        {
            for (const Train::StepOutput& output : step.outputs)
            {
                if (output.name == name)
                    return output.tensor;
            }

            return nullptr;
        }
    } // namespace

    Result<std::unique_ptr<ExpertBalanceObjective>> ExpertBalanceObjective::make(
        std::vector<std::string> balances, std::vector<ExpertUsageOutput> usages)
    {
        SA_FAIL_IF(balances.empty() || usages.empty(), StatusCode::InvalidDescription,
                   "An expert_balance objective reads at least one balance output and one "
                   "usage output; it was given {} and {}",
                   balances.size(), usages.size());

        for (const std::string& balance : balances)
        {
            SA_FAIL_IF(balance.empty(), StatusCode::InvalidDescription,
                       "An expert_balance objective was given a balance output with no "
                       "name among its {}",
                       balances.size());
        }

        for (const ExpertUsageOutput& usage : usages)
        {
            SA_FAIL_IF(usage.name.empty() || usage.experts < 1, StatusCode::InvalidDescription,
                       "An expert_balance objective reads a named usage of at least one "
                       "expert; it was given '{}' of {}",
                       usage.name, usage.experts);
        }

        return SushiAI::Detail::guard_value(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            return std::unique_ptr<ExpertBalanceObjective>(
                new ExpertBalanceObjective(std::move(balances), std::move(usages)));
        });
    }

    ExpertBalanceObjective::ExpertBalanceObjective(
        std::vector<std::string> balances, std::vector<ExpertUsageOutput> usages) noexcept
        : balances_(std::move(balances)), usages_(std::move(usages))
    {

    }

    void ExpertBalanceObjective::declare(const Graph::GraphBuilder&,
                                         Train::ObjectiveLeaves&) const
    {

    }

    Result<Graph::ValueID> ExpertBalanceObjective::loss(
        const Graph::GraphBuilder& builder, const Graph::Graph& graph,
        const NN::ModelOutputs& outputs, const Train::ObjectiveLeaves&) const
    {
        Graph::ValueID total = Graph::K_NO_VALUE;
        for (const std::string& name : balances_)
        {
            const Result<Graph::ValueID> found = outputs.at(name);
            SA_FATAL_IF(!found.has_value(), "The expert_balance objective reads '{}': {}",
                        name, found.status().message());

            // By value: every emission below can reallocate the value vector.
            const Shape shape = graph.value(found.value()).shape;
            SA_FATAL_IF(shape.num_elements() != 1,
                        "The expert_balance objective reads '{}' as one element; the model "
                        "published {}",
                        name, shape.to_string());

            const Graph::ValueID term = builder.reshape(found.value(), Shape{1});
            total = total == Graph::K_NO_VALUE ? term : builder.add(total, term);
        }

        // A mean over one term changes nothing, so it emits no node.
        if (balances_.size() == 1)
            return total;

        return builder.scale(total, 1.0 / static_cast<double>(balances_.size()));
    }

    Status ExpertBalanceObjective::feed(const Data::Batch&, const Train::ObjectiveLeaves&,
                                        Graph::TensorPool&) const
    {
        return Status();
    }

    Status ExpertBalanceObjective::measure(const Train::StepView& step,
                                           Train::Metrics& metrics) const
    {
        double smallest = 1.0;
        double largest = 0.0;
        for (const ExpertUsageOutput& usage : usages_)
        {
            const SushiBLAS::Tensor* const shares = find_step_output(step, usage.name);
            SA_FAIL_IF(shares == nullptr, StatusCode::InvalidDescription,
                       "The expert_balance objective measures the output '{}', which the "
                       "step did not publish among its {}",
                       usage.name, step.outputs.size());
            SA_TRY(Train::require_host_reals(*shares, usage.experts, "the usage of the experts"));

            for (std::int64_t expert = 0; expert < usage.experts; ++expert)
            {
                const double share = Train::host_real(*shares, expert);
                smallest = std::min(smallest, share);
                largest = std::max(largest, share);
            }
        }

        SA_TRY(metrics.add(Train::Metric{K_EXPERT_USAGE_MIN_METRIC, smallest, true}));

        return metrics.add(Train::Metric{K_EXPERT_USAGE_MAX_METRIC, largest, true});
    }

    Result<std::unique_ptr<Train::Objective>> make_expert_balance_objective(
        const Train::ObjectiveContext& given, const Config::Value& attributes,
        std::string_view context)
    {
        SA_FAIL_IF(!attributes.keys().empty(), StatusCode::InvalidDescription,
                   "{} states the key '{}', but an expert_balance objective has no keys",
                   context, attributes.keys().front());

        std::vector<std::string> balances;
        std::vector<ExpertUsageOutput> usages;
        SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]() -> Status
        {
            for (const Train::OutputShape& output : given.outputs)
            {
                if (begins_with(output.name, K_BALANCE_OUTPUT_PREFIX))
                {
                    SA_FAIL_IF(output.shape.num_elements() != 1,
                               StatusCode::InvalidDescription,
                               "{} reads the balance '{}' as one element, but the model "
                               "publishes it as {}",
                               context, output.name, output.shape.to_string());
                    balances.push_back(output.name);
                }
                else if (begins_with(output.name, K_USAGE_OUTPUT_PREFIX))
                {
                    usages.push_back(
                        ExpertUsageOutput{output.name, output.shape.num_elements()});
                }
            }

            return Status();
        }));

        SA_FAIL_IF(balances.empty() || usages.empty(), StatusCode::InvalidDescription,
                   "{} reads the outputs whose names begin with '{}' and with '{}', but the "
                   "model publishes {} and {} of them among its {}",
                   context, K_BALANCE_OUTPUT_PREFIX, K_USAGE_OUTPUT_PREFIX, balances.size(),
                   usages.size(), given.outputs.size());

        SA_TRY_ASSIGN(std::unique_ptr<ExpertBalanceObjective> made,
                      ExpertBalanceObjective::make(std::move(balances), std::move(usages)));
        std::unique_ptr<Train::Objective> base = std::move(made);
        return std::move(base);
    }

} // namespace SushiAI::Language::Training
