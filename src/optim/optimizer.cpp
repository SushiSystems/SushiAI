/**************************************************************************/
/* optimizer.cpp                                                          */
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
 * @file optimizer.cpp
 * @brief Implements abstract base optimizer methods and parameter slot bindings.
 * @author Mustafa Garip
 */

#include <SushiAI/optim/optimizer.hpp>
#include <cstddef>
#include <utility>
#include <vector>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/seam.hpp>

namespace SushiAI
{
    namespace Optim
    {
        namespace
        {
            /**
             * @brief Lists the address of every slot's gradient, in slot order.
             * @param slots Slots whose gradients are listed; must outlive the list.
             * @return The addresses, or the failure of the allocation seam.
             */
            Result<std::vector<SushiBLAS::Tensor*>> list_gradients(
                std::vector<ParameterSlot>& slots)
            {
                return Detail::guard_value(Detail::SeamPhase::Allocate, [&]()
                {
                    std::vector<SushiBLAS::Tensor*> gradients;
                    gradients.reserve(slots.size());
                    for (ParameterSlot& slot : slots) gradients.push_back(&slot.gradient);
                    return gradients;
                });
            }
        }

        Status validate_slots(SushiRuntime::span<const ParameterSlot> slots)
        {
            for (const ParameterSlot& slot : slots)
            {
                SA_FAIL_IF(slot.parameter.num_elements != slot.gradient.num_elements,
                           StatusCode::InvalidDescription,
                           "Parameter '{}' has {} elements but its gradient has {}",
                           slot.name, slot.parameter.num_elements, slot.gradient.num_elements);
                SA_FAIL_IF(slot.parameter.dtype != slot.gradient.dtype,
                           StatusCode::InvalidDescription,
                           "Parameter '{}' is {} but its gradient is {}",
                           slot.name, dtype_name(slot.parameter.dtype),
                           dtype_name(slot.gradient.dtype));
                SA_FAIL_IF(!is_real_float(slot.parameter.dtype), StatusCode::InvalidDescription,
                           "Parameter '{}' is {}; optimizers update real floating tensors only",
                           slot.name, dtype_name(slot.parameter.dtype));
                SA_FAIL_IF(slot.parameter.dtype == DType::Half, StatusCode::InvalidDescription,
                           "Parameter '{}' is half precision; SushiBLAS's RNG cannot initialise "
                           "half state and a half accumulator loses the small updates that make "
                           "momentum work. Keep parameters in float32.",
                           slot.name);
            }

            return Status();
        }

        Optimizer::Optimizer(SushiBLAS::Engine& engine,
                             std::vector<ParameterSlot> parameters) noexcept
            : engine_(&engine), parameters_(std::move(parameters))
        {

        }

        Status Optimizer::set_loss_scaler(LossScaler* scaler)
        {
            SA_FAIL_IF(scaler != nullptr && !is_replay_invariant(),
                       StatusCode::InvalidDescription,
                       "This optimizer records its update as separate operations, and a "
                       "skipped step has to be a branch taken inside one kernel. The separate "
                       "spelling has no branch to take: it would unscale the gradient "
                       "correctly and then apply the overflowed step anyway, converging on "
                       "whatever an infinity does to the moments. Use "
                       "Optim::UpdateKernel::Fused with loss scaling.");
            SA_FAIL_IF(scaler != nullptr && scaler->parameter_count() != parameters_.size(),
                       StatusCode::InvalidDescription,
                       "This loss scaler screens {} gradients but the optimizer updates {} "
                       "parameters; the gate's slot index is a position in the optimizer's "
                       "own slot list, so the two must be the same length",
                       scaler->parameter_count(), parameters_.size());

            scaler_ = scaler;
            return Status();
        }

        float Optimizer::inverse_loss_scale() const noexcept
        {
            return scaler_ == nullptr ? 1.0f : static_cast<float>(scaler_->inverse_scale());
        }

        const std::int64_t* Optimizer::gradient_gate() const noexcept
        {
            return scaler_ == nullptr ? nullptr : scaler_->gate();
        }

        void Optimizer::before_replay()
        {
            if (scaler_ != nullptr) scaler_->before_replay();
            write_step_scalars();
        }

        void Optimizer::after_replay()
        {
            if (scaler_ != nullptr)
            {
                scaler_->after_replay();

                // Retain step counter without advancing when current step was skipped.
                if (scaler_->last_step_skipped()) return;
            }

            advance();
        }

        Result<std::size_t> Optimizer::record_all(SushiBLAS::TaskRecorder& recorder)
        {
            // Record loss scaler gradient screens before recording parameter updates.
            std::size_t recorded = 0;

            if (scaler_ != nullptr)
            {
                SA_TRY_ASSIGN(const std::vector<SushiBLAS::Tensor*> gradients,
                              list_gradients(parameters_));
                SA_TRY_ASSIGN(const std::size_t screens,
                              scaler_->record_screens(recorder, gradients));
                recorded += screens;
            }

            // record_update() may throw; this guard is the only place its exception is caught.
            SA_TRY_ASSIGN(const std::size_t updates,
                          Detail::guard_value(Detail::SeamPhase::Record,
                                              [&]() { return record_update(recorder); }));
            return recorded + updates;
        }

        Result<std::size_t> Optimizer::record_into_step(SushiBLAS::TaskRecorder& recorder)
        {
            // Mark step recording as owned by compiled step execution graph.
            in_compiled_step_ = true;
            return record_all(recorder);
        }

        Status Optimizer::step()
        {
            SA_FATAL_IF(in_compiled_step_,
                        "This optimizer's update is already recorded into a compiled step, "
                        "which applies it on every replay. Calling step() as well would apply "
                        "it twice per batch - the loop would converge at double the learning "
                        "rate and report nothing wrong. Let the step drive it, or build the "
                        "trainer with OptimizerPlacement::PerStep.");

            // Execute three-phase update sequence matching compiled plan replay order.
            before_replay();
            SA_TRY(Detail::record_and_run(*engine_, [this]() -> Status
            {
                SA_TRY(record_all(engine_->get_recorder()));
                return Status();
            }));
            after_replay();
            return Status();
        }

        Result<SushiBLAS::Tensor> Optimizer::allocate_state(const SushiBLAS::Tensor& like) const
        {
            // Allocate distinct tensor storage matching target dtype and shape layout.
            return allocate_like(*engine_, like);
        }

    } // namespace Optim
} // namespace SushiAI
