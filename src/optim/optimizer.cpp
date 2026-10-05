/****************************************************************/
/* optimizer.cpp                                                */
/* SushiAI - https://github.com/SushiSystems/SushiAI            */
/* Copyright (c) 2026 Sushi Systems                             */
/* Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.    */
/* Commercial use requires a licence from Sushi Systems.        */
/****************************************************************/

/**
 * @file optimizer.cpp
 * @brief Implements abstract base optimizer methods and parameter slot bindings.
 * @author Mustafa Garip
 */

#include <utility>
#include <vector>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/optim/optimizer.hpp>

namespace SushiAI
{
    namespace Optim
    {
        Optimizer::Optimizer(SushiBLAS::Engine& engine, std::vector<ParameterSlot> parameters)
            : engine_(&engine), parameters_(std::move(parameters))
        {
            for (const ParameterSlot& slot : parameters_)
            {
                SA_THROW_IF(slot.parameter.num_elements != slot.gradient.num_elements,
                            "Parameter '{}' has {} elements but its gradient has {}",
                            slot.name, slot.parameter.num_elements, slot.gradient.num_elements);
                SA_THROW_IF(slot.parameter.dtype != slot.gradient.dtype,
                            "Parameter '{}' is {} but its gradient is {}",
                            slot.name, dtype_name(slot.parameter.dtype),
                            dtype_name(slot.gradient.dtype));
                SA_THROW_IF(!is_real_float(slot.parameter.dtype),
                            "Parameter '{}' is {}; optimizers update real floating tensors only",
                            slot.name, dtype_name(slot.parameter.dtype));
                SA_THROW_IF(slot.parameter.dtype == DType::HALF,
                            "Parameter '{}' is half precision; SushiBLAS's RNG cannot initialise "
                            "half state and a half accumulator loses the small updates that make "
                            "momentum work. Keep parameters in float32.",
                            slot.name);
            }
        }

        void Optimizer::set_loss_scaler(LossScaler* scaler)
        {
            SA_THROW_IF(scaler != nullptr && !is_replay_invariant(),
                        "This optimizer records its update as separate operations, and a "
                        "skipped step has to be a branch taken inside one kernel. The separate "
                        "spelling has no branch to take: it would unscale the gradient "
                        "correctly and then apply the overflowed step anyway, converging on "
                        "whatever an infinity does to the moments. Use "
                        "Optim::UpdateKernel::FUSED with loss scaling.");
            SA_THROW_IF(scaler != nullptr && scaler->parameter_count() != parameters_.size(),
                        "This loss scaler screens {} gradients but the optimizer updates {} "
                        "parameters; the gate's slot index is a position in the optimizer's "
                        "own slot list, so the two must be the same length",
                        scaler->parameter_count(), parameters_.size());

            scaler_ = scaler;
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

        std::size_t Optimizer::record_all()
        {
            // Record loss scaler gradient screens before recording parameter updates.
            std::size_t recorded = 0;

            if (scaler_ != nullptr)
            {
                std::vector<SushiBLAS::Tensor*> gradients;
                gradients.reserve(parameters_.size());
                for (ParameterSlot& slot : parameters_) gradients.push_back(&slot.gradient);

                recorded += scaler_->record_screens(gradients);
            }

            return recorded + record_update();
        }

        std::size_t Optimizer::record_into_step()
        {
            // Mark step recording as owned by compiled step execution graph.
            in_compiled_step_ = true;
            return record_all();
        }

        void Optimizer::step()
        {
            SA_THROW_IF(in_compiled_step_,
                        "This optimizer's update is already recorded into a compiled step, "
                        "which applies it on every replay. Calling step() as well would apply "
                        "it twice per batch - the loop would converge at double the learning "
                        "rate and report nothing wrong. Let the step drive it, or build the "
                        "trainer with OptimizerPlacement::PER_STEP.");

            // Execute three-phase update sequence matching compiled plan replay order.
            before_replay();
            (void)record_all();
            flush();
            after_replay();
        }

        void Optimizer::flush()
        {
            engine_->execute();
            engine_->get_context().wait_all();
        }

        SushiBLAS::Tensor Optimizer::allocate_state(const SushiBLAS::Tensor& like) const
        {
            // Allocate distinct tensor storage matching target dtype and shape layout.
            return allocate_like(*engine_, like);
        }

    } // namespace Optim
} // namespace SushiAI
