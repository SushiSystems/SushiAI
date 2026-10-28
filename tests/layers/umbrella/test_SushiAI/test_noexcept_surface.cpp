/**************************************************************************/
/* test_noexcept_surface.cpp                                              */
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
 * @file test_noexcept_surface.cpp
 * @brief Tests that every public function of the core and layers modules is noexcept.
 * @author Mustafa Garip
 */

#include <utility>

#include <gtest/gtest.h>

#include <SushiAI/allocation/allocation.hpp>
#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/gradcheck.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/autograd/rules/activation.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/autograd/rules/elementwise.hpp>
#include <SushiAI/autograd/rules/linalg.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/autograd/rules/norm.hpp>
#include <SushiAI/autograd/rules/random.hpp>
#include <SushiAI/autograd/rules/shape.hpp>
#include <SushiAI/autograd/rules/spatial.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/config/data_fit.hpp>
#include <SushiAI/config/json_events.hpp>
#include <SushiAI/config/json_writer.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/model_writer.hpp>
#include <SushiAI/config/section_reader.hpp>
#include <SushiAI/config/train_config.hpp>
#include <SushiAI/config/value.hpp>
#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/epoch.hpp>
#include <SushiAI/data/epoch_permutation.hpp>
#include <SushiAI/data/materialised_dataset.hpp>
#include <SushiAI/data/target.hpp>
#include <SushiAI/data/threaded_loader.hpp>
#include <SushiAI/dtype/dtype.hpp>
#include <SushiAI/eager/region_boundary.hpp>
#include <SushiAI/eager/region_emitter.hpp>
#include <SushiAI/eager/region_executor.hpp>
#include <SushiAI/eager/region_names.hpp>
#include <SushiAI/eager/region_program.hpp>
#include <SushiAI/eager/region_signature.hpp>
#include <SushiAI/eager/region_table.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/session.hpp>
#include <SushiAI/eager/slot_ledger.hpp>
#include <SushiAI/eager/step.hpp>
#include <SushiAI/eager/storage_bank.hpp>
#include <SushiAI/eager/storage_id.hpp>
#include <SushiAI/engine_fixture/test_environment.hpp>
#include <SushiAI/engine_fixture/test_graph.hpp>
#include <SushiAI/failure/check.hpp>
#include <SushiAI/failure/status.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/dropout_mask.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/lowering/activation.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/graph/lowering/elementwise.hpp>
#include <SushiAI/graph/lowering/fused.hpp>
#include <SushiAI/graph/lowering/linalg.hpp>
#include <SushiAI/graph/lowering/loss.hpp>
#include <SushiAI/graph/lowering/norm.hpp>
#include <SushiAI/graph/lowering/random.hpp>
#include <SushiAI/graph/lowering/shape.hpp>
#include <SushiAI/graph/lowering/spatial.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/rematerialise.hpp>
#include <SushiAI/graph/state_id.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/step_recording.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/hash/fnv1a.hpp>
#include <SushiAI/io/checkpoint.hpp>
#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/imported_model_writer.hpp>
#include <SushiAI/io/importer_registry.hpp>
#include <SushiAI/io/run_checkpoint.hpp>
#include <SushiAI/io/run_progress.hpp>
#include <SushiAI/nn/add.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/built_rows.hpp>
#include <SushiAI/nn/channel_slice.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/depthwise_conv2d.hpp>
#include <SushiAI/nn/dropout.hpp>
#include <SushiAI/nn/embedding.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/host_table.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/layer_norm.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/mish.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include <SushiAI/ops/ops.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/apply_step_schedule.hpp>
#include <SushiAI/optim/cosine_schedule.hpp>
#include <SushiAI/optim/cosine_schedule_section.hpp>
#include <SushiAI/optim/gradient_clip.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>
#include <SushiAI/optim/rate_schedule.hpp>
#include <SushiAI/optim/rate_schedule_section.hpp>
#include <SushiAI/optim/run_schedule.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/step_scalars.hpp>
#include <SushiAI/optim/step_schedule.hpp>
#include <SushiAI/optim/step_schedule_section.hpp>
#include <SushiAI/profile/timelines.hpp>
#include <SushiAI/profile/zone_groups.hpp>
#include <SushiAI/shape/shape.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/train/batch_role.hpp>
#include <SushiAI/train/binary_confusion.hpp>
#include <SushiAI/train/binary_cross_entropy_objective.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/decision_threshold.hpp>
#include <SushiAI/train/dense_losses.hpp>
#include <SushiAI/train/dense_targets.hpp>
#include <SushiAI/train/evaluation_step.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/host_values.hpp>
#include <SushiAI/train/input_match.hpp>
#include <SushiAI/train/mean_squared_error_objective.hpp>
#include <SushiAI/train/metrics.hpp>
#include <SushiAI/train/multi_label_objective.hpp>
#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/objective_context.hpp>
#include <SushiAI/train/objective_registry.hpp>
#include <SushiAI/train/one_hot.hpp>
#include <SushiAI/train/parameter_groups.hpp>
#include <SushiAI/train/role_match.hpp>
#include <SushiAI/train/step_view.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/sum_objective.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/version.hpp>

#if defined(SA_TEST_DISTRIBUTED_BUILD)
#include <SushiAI/distribution/cluster_section.hpp>
#include <SushiAI/distribution/exact_sum.hpp>
#include <SushiAI/distribution/gradient_exchange.hpp>
#include <SushiAI/distribution/host_access.hpp>
#include <SushiAI/distribution/parameter_publish.hpp>
#include <SushiAI/distribution/replica_draws.hpp>
#include <SushiAI/distribution/replica_kernel.hpp>
#include <SushiAI/distribution/replica_plan.hpp>
#include <SushiAI/distribution/replica_position.hpp>
#include <SushiAI/distribution/replica_step.hpp>
#include <SushiAI/distribution/replica_task.hpp>
#endif

namespace SushiAI
{
    static_assert(
        noexcept(allocate_tensor(std::declval<SushiBLAS::Engine&>(),
                std::declval<SushiRuntime::span<const int64_t>>(), std::declval<DType>())),
        "allocate_tensor is noexcept");
    static_assert(
        noexcept(allocate_tensor(std::declval<SushiBLAS::Engine&>(), std::declval<const Shape&>(),
                std::declval<DType>())),
        "allocate_tensor is noexcept");
    static_assert(
        noexcept(allocate_like(std::declval<SushiBLAS::Engine&>(),
                std::declval<const SushiBLAS::Tensor&>())),
        "allocate_like is noexcept");
    static_assert(
        noexcept(allocate_arena(std::declval<SushiBLAS::Engine&>(), std::declval<std::size_t>())),
        "allocate_arena is noexcept");
    static_assert(
        noexcept(view_tensor(std::declval<SushiBLAS::Engine&>(),
                std::declval<const SushiRuntime::sushi_ptr<SushiBLAS::Storage>&>(),
                std::declval<std::size_t>(), std::declval<const Shape&>(), std::declval<DType>())),
        "view_tensor is noexcept");
} // namespace SushiAI

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(has_backward_rule(std::declval<OpID>())),
        "has_backward_rule is noexcept");
    static_assert(
        noexcept(differentiate(std::declval<Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<SushiRuntime::span<const Graph::ValueID>>(),
                std::declval<Graph::ValueID>(), std::declval<Provenance*>())),
        "differentiate is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(std::declval<const GradCheckResult&>().to_string()),
        "to_string is noexcept");
    static_assert(
        noexcept(push_off_kinks(std::declval<SushiRuntime::span<double>>(),
                std::declval<const GradCheckOptions&>())),
        "push_off_kinks is noexcept");
    static_assert(
        noexcept(check_full_jacobian(std::declval<const ScalarFunction&>(),
                std::declval<const GradientFunction&>(), std::declval<SushiRuntime::span<double>>(),
                std::declval<const GradCheckOptions&>())),
        "check_full_jacobian is noexcept");
    static_assert(
        noexcept(check_directional(std::declval<const ScalarFunction&>(),
                std::declval<const GradientFunction&>(), std::declval<SushiRuntime::span<double>>(),
                std::declval<std::size_t>(), std::declval<const GradCheckOptions&>())),
        "check_directional is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(std::declval<const Provenance&>().origin_of(std::declval<Graph::NodeID>())),
        "origin_of is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(activation_backward(std::declval<const GradientTape&>())),
        "activation_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(noexcept(has_rule(std::declval<OpID>())), "has_rule is noexcept");
    static_assert(
        noexcept(every_differentiable_op_has_a_rule(
                std::declval<SushiRuntime::span<const Ops::OpName>>())),
        "every_differentiable_op_has_a_rule is noexcept");
    static_assert(
        noexcept(no_gradient_barrier_has_a_rule(std::declval<SushiRuntime::span<const Ops::OpName>>(
                ))),
        "no_gradient_barrier_has_a_rule is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(add_backward(std::declval<const GradientTape&>())),
        "add_backward is noexcept");
    static_assert(
        noexcept(sub_backward(std::declval<const GradientTape&>())),
        "sub_backward is noexcept");
    static_assert(
        noexcept(mul_backward(std::declval<const GradientTape&>())),
        "mul_backward is noexcept");
    static_assert(
        noexcept(scale_backward(std::declval<const GradientTape&>())),
        "scale_backward is noexcept");
    static_assert(
        noexcept(exp_backward(std::declval<const GradientTape&>())),
        "exp_backward is noexcept");
    static_assert(
        noexcept(accumulate_backward(std::declval<const GradientTape&>())),
        "accumulate_backward is noexcept");
    static_assert(
        noexcept(div_backward(std::declval<const GradientTape&>())),
        "div_backward is noexcept");
    static_assert(
        noexcept(minimum_backward(std::declval<const GradientTape&>())),
        "minimum_backward is noexcept");
    static_assert(
        noexcept(maximum_backward(std::declval<const GradientTape&>())),
        "maximum_backward is noexcept");
    static_assert(
        noexcept(abs_backward(std::declval<const GradientTape&>())),
        "abs_backward is noexcept");
    static_assert(
        noexcept(atan_backward(std::declval<const GradientTape&>())),
        "atan_backward is noexcept");
    static_assert(
        noexcept(select_backward(std::declval<const GradientTape&>())),
        "select_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(matmul_backward(std::declval<const GradientTape&>())),
        "matmul_backward is noexcept");
    static_assert(
        noexcept(bias_add_backward(std::declval<const GradientTape&>())),
        "bias_add_backward is noexcept");
    static_assert(
        noexcept(sum_rows_backward(std::declval<const GradientTape&>())),
        "sum_rows_backward is noexcept");
    static_assert(
        noexcept(gather_rows_backward(std::declval<const GradientTape&>())),
        "gather_rows_backward is noexcept");
    static_assert(
        noexcept(scatter_add_rows_backward(std::declval<const GradientTape&>())),
        "scatter_add_rows_backward is noexcept");
    static_assert(
        noexcept(causal_mask_backward(std::declval<const GradientTape&>())),
        "causal_mask_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(log_softmax_backward(std::declval<const GradientTape&>())),
        "log_softmax_backward is noexcept");
    static_assert(
        noexcept(nll_loss_backward(std::declval<const GradientTape&>())),
        "nll_loss_backward is noexcept");
    static_assert(
        noexcept(starts_from_implicit_seed(std::declval<OpID>())),
        "starts_from_implicit_seed is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(batchnorm2d_backward(std::declval<const GradientTape&>())),
        "batchnorm2d_backward is noexcept");
    static_assert(
        noexcept(channel_affine_backward(std::declval<const GradientTape&>())),
        "channel_affine_backward is noexcept");
    static_assert(
        noexcept(layer_norm_backward(std::declval<const GradientTape&>())),
        "layer_norm_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(dropout_backward(std::declval<const GradientTape&>())),
        "dropout_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(reshape_backward(std::declval<const GradientTape&>())),
        "reshape_backward is noexcept");
    static_assert(
        noexcept(layout_convert_backward(std::declval<const GradientTape&>())),
        "layout_convert_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(
        noexcept(im2col_backward(std::declval<const GradientTape&>())),
        "im2col_backward is noexcept");
    static_assert(
        noexcept(col2im_backward(std::declval<const GradientTape&>())),
        "col2im_backward is noexcept");
    static_assert(
        noexcept(maxpool2d_backward(std::declval<const GradientTape&>())),
        "maxpool2d_backward is noexcept");
    static_assert(
        noexcept(avgpool2d_backward(std::declval<const GradientTape&>())),
        "avgpool2d_backward is noexcept");
    static_assert(
        noexcept(upsample2d_backward(std::declval<const GradientTape&>())),
        "upsample2d_backward is noexcept");
    static_assert(
        noexcept(spatial_mean_backward(std::declval<const GradientTape&>())),
        "spatial_mean_backward is noexcept");
    static_assert(
        noexcept(concat_channels_backward(std::declval<const GradientTape&>())),
        "concat_channels_backward is noexcept");
    static_assert(
        noexcept(slice_channels_backward(std::declval<const GradientTape&>())),
        "slice_channels_backward is noexcept");
    static_assert(
        noexcept(depthwise_conv2d_backward(std::declval<const GradientTape&>())),
        "depthwise_conv2d_backward is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Autograd
{
    static_assert(noexcept(std::declval<const GradientTape&>().graph()), "graph is noexcept");
    static_assert(noexcept(std::declval<const GradientTape&>().builder()), "builder is noexcept");
    static_assert(noexcept(std::declval<const GradientTape&>().node()), "node is noexcept");
    static_assert(noexcept(std::declval<const GradientTape&>().node_id()), "node_id is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().input(std::declval<std::size_t>())),
        "input is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().output(std::declval<std::size_t>())),
        "output is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().output_grad(std::declval<std::size_t>())),
        "output_grad is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().output_is_root(std::declval<std::size_t>())),
        "output_is_root is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().input_requires_grad(std::declval<std::size_t>(
                ))),
        "input_requires_grad is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().contribute(std::declval<Graph::ValueID>(),
                std::declval<const Graph::Traced&>())),
        "contribute is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().contribute_input(std::declval<std::size_t>(),
                std::declval<const Graph::Traced&>())),
        "contribute_input is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().observe(std::declval<const Status&>())),
        "observe is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().skip(std::declval<Graph::NodeID>())),
        "skip is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().seed_factor()),
        "seed_factor is noexcept");
    static_assert(
        noexcept(std::declval<const GradientTape&>().apply_seed_factor(
                std::declval<const Graph::Traced&>())),
        "apply_seed_factor is noexcept");
    static_assert(
        noexcept(GradientTape(std::declval<Graph::GraphBuilder>(), std::declval<Differentiator&>(),
                std::declval<Graph::ValueID>())),
        "GradientTape is noexcept");
    static_assert(
        noexcept(std::declval<GradientTape&>().retarget(std::declval<Graph::NodeID>())),
        "retarget is noexcept");
} // namespace SushiAI::Autograd

namespace SushiAI::Config
{
    static_assert(
        noexcept(fit_data_to_model(std::declval<const Value&>(), std::declval<const ModelSpec&>(),
                std::declval<std::string_view>())),
        "fit_data_to_model is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(read_json_events(std::declval<const std::filesystem::path&>(),
                std::declval<JSONEvents&>())),
        "read_json_events is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(write_json(std::declval<const Value&>(), std::declval<std::size_t>())),
        "write_json is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(parse_model(std::declval<const Value&>(), std::declval<std::string_view>())),
        "parse_model is noexcept");
    static_assert(
        noexcept(load_model(std::declval<const std::filesystem::path&>())),
        "load_model is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(write_model(std::declval<const ModelSpec&>())),
        "write_model is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(member_of(std::declval<std::string_view>(), std::declval<std::string_view>())),
        "member_of is noexcept");
    static_assert(
        noexcept(entry_of(std::declval<std::string_view>(), std::declval<std::size_t>())),
        "entry_of is noexcept");
    static_assert(noexcept(list_quoted(std::declval<KeySpan>())), "list_quoted is noexcept");
    static_assert(
        noexcept(read_real(std::declval<const Value&>(), std::declval<std::string_view>())),
        "read_real is noexcept");
    static_assert(
        noexcept(read_whole(std::declval<const Value&>(), std::declval<std::string_view>())),
        "read_whole is noexcept");
    static_assert(
        noexcept(read_float32(std::declval<const Value&>(), std::declval<std::string_view>())),
        "read_float32 is noexcept");
    static_assert(
        noexcept(read_string(std::declval<const Value&>(), std::declval<std::string_view>())),
        "read_string is noexcept");
    static_assert(
        noexcept(read_array(std::declval<const Value&>(), std::declval<std::string_view>())),
        "read_array is noexcept");
    static_assert(
        noexcept(SectionReader(std::declval<const Value&>(), std::declval<std::string_view>())),
        "SectionReader is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().check_object()),
        "check_object is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().check_keys(std::declval<KeySpan>())),
        "check_keys is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().check_type(std::declval<std::string_view>(),
                std::declval<std::string_view>(), std::declval<std::string_view>())),
        "check_type is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().has(std::declval<std::string_view>())),
        "has is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().member(std::declval<std::string_view>())),
        "member is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().find(std::declval<std::string_view>())),
        "find is noexcept");
    static_assert(
        noexcept(std::declval<const SectionReader&>().find_of(std::declval<std::string_view>(),
                std::declval<Kind>())),
        "find_of is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(
        noexcept(parse_train(std::declval<const Value&>(), std::declval<std::string_view>(),
                std::declval<const std::filesystem::path&>())),
        "parse_train is noexcept");
    static_assert(
        noexcept(load_train(std::declval<const std::filesystem::path&>())),
        "load_train is noexcept");
    static_assert(
        noexcept(check_start(std::declval<const TrainSpec&>(), std::declval<std::string_view>())),
        "check_start is noexcept");
    static_assert(
        noexcept(read_run_seed(std::declval<const TrainSpec&>(), std::declval<std::string_view>())),
        "read_run_seed is noexcept");
    static_assert(
        noexcept(parse_data(std::declval<const Value&>(), std::declval<std::string_view>(),
                            std::declval<const std::filesystem::path&>())),
        "parse_data is noexcept");
    static_assert(
        noexcept(load_data(std::declval<const std::filesystem::path&>())),
        "load_data is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Config
{
    static_assert(noexcept(kind_name(std::declval<Kind>())), "kind_name is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().boolean(std::declval<bool>())),
        "boolean is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().integer(std::declval<std::int64_t>())),
        "integer is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().real(std::declval<double>())),
        "real is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().string(std::declval<std::string>())),
        "string is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().array(std::declval<std::vector<Value>>())),
        "array is noexcept");
    static_assert(
        noexcept(std::declval<Value&>().object(std::declval<std::vector<std::string>>(),
                std::declval<std::vector<Value>>())),
        "object is noexcept");
    static_assert(noexcept(std::declval<const Value&>().kind()), "kind is noexcept");
    static_assert(noexcept(std::declval<const Value&>().is_null()), "is_null is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().as_bool(std::declval<std::string_view>())),
        "as_bool is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().as_int(std::declval<std::string_view>())),
        "as_int is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().as_real(std::declval<std::string_view>())),
        "as_real is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().as_string(std::declval<std::string_view>())),
        "as_string is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().as_array(std::declval<std::string_view>())),
        "as_array is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().has(std::declval<std::string_view>())),
        "has is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().at(std::declval<std::string_view>(),
                std::declval<std::string_view>())),
        "at is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().at_or(std::declval<std::string_view>(),
                std::declval<const Value&>())),
        "at_or is noexcept");
    static_assert(noexcept(std::declval<const Value&>().keys()), "keys is noexcept");
    static_assert(noexcept(std::declval<const Value&>().values()), "values is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().reject_unknown_keys(std::declval<std::string_view>(),
                std::declval<SushiRuntime::span<const std::string_view>>())),
        "reject_unknown_keys is noexcept");
    static_assert(
        noexcept(std::declval<const Value&>().without(std::declval<std::string_view>())),
        "without is noexcept");
    static_assert(
        noexcept(require_version(std::declval<const Value&>(), std::declval<std::string_view>(),
                std::declval<std::int64_t>(), std::declval<std::string_view>())),
        "require_version is noexcept");
    static_assert(
        noexcept(parse_json(std::declval<std::string_view>(), std::declval<std::string_view>())),
        "parse_json is noexcept");
    static_assert(
        noexcept(parse_json_file(std::declval<const std::filesystem::path&>())),
        "parse_json_file is noexcept");
} // namespace SushiAI::Config

namespace SushiAI::Data
{
    static_assert(
        noexcept(describe_input(std::declval<const Dataset&>())),
        "describe_input is noexcept");
    static_assert(
        noexcept(check_rows(std::declval<const Dataset&>(), std::declval<std::size_t>(),
                std::declval<RowRange>())),
        "check_rows is noexcept");
    static_assert(
        noexcept(std::declval<SyntheticClassification&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SyntheticOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<Rings&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<RingsOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(make_rings(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_rings is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(std::declval<DatasetRegistry&>().add(std::declval<const DatasetEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const DatasetRegistry&>().find(std::declval<std::string_view>())),
        "find is noexcept");
    static_assert(noexcept(std::declval<const DatasetRegistry&>().rows()), "rows is noexcept");
    static_assert(
        noexcept(register_datasets(std::declval<DatasetRegistry&>())),
        "register_datasets is noexcept");
    static_assert(
        noexcept(make_dataset(std::declval<const DatasetRegistry&>(),
                std::declval<SushiBLAS::Engine&>(), std::declval<std::string_view>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_dataset is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(begin_epoch(std::declval<const Dataset&>(), std::declval<std::int64_t>())),
        "begin_epoch is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(EpochPermutation(std::declval<std::uint64_t>(), std::declval<std::int64_t>(),
                std::declval<std::uint64_t>())),
        "EpochPermutation is noexcept");
    static_assert(
        noexcept(std::declval<const EpochPermutation&>().at(std::declval<std::uint64_t>())),
        "at is noexcept");
    static_assert(noexcept(std::declval<const EpochPermutation&>().count()), "count is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(MaterialisedDataset(std::declval<std::vector<Batch>>(), std::declval<int64_t>(),
                std::declval<const InputDescription&>(),
                std::declval<std::vector<TargetDescription>>(), std::declval<std::string>())),
        "MaterialisedDataset is noexcept");
    static_assert(
        noexcept(allocate_batch(std::declval<SushiBLAS::Engine&>(), std::declval<int64_t>(),
                std::declval<const InputDescription&>(),
                std::declval<const std::vector<TargetDescription>&>())),
        "allocate_batch is noexcept");
    static_assert(
        noexcept(share_batch(std::declval<const Batch&>(), std::declval<Batch&>())),
        "share_batch is noexcept");
    static_assert(
        noexcept(share_rows(std::declval<const Batch&>(), std::declval<RowRange>(),
                std::declval<Batch&>())),
        "share_rows is noexcept");
    static_assert(
        noexcept(prepare_batch(std::declval<SushiBLAS::Engine&>(), std::declval<int64_t>(),
                std::declval<const InputDescription&>(),
                std::declval<const std::vector<TargetDescription>&>(), std::declval<Batch&>())),
        "prepare_batch is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(find_target(std::declval<const std::vector<BatchTarget>&>(),
                std::declval<std::string_view>())),
        "find_target is noexcept");
    static_assert(
        noexcept(find_target(std::declval<const std::vector<TargetDescription>&>(),
                std::declval<std::string_view>())),
        "find_target is noexcept");
} // namespace SushiAI::Data

namespace SushiAI::Data
{
    static_assert(
        noexcept(std::declval<ThreadedLoader&>().make(std::declval<std::unique_ptr<Dataset>>(),
                std::declval<const LoaderOptions&>())),
        "make is noexcept");
} // namespace SushiAI::Data

namespace SushiAI
{
    static_assert(noexcept(dtype_size(std::declval<DType>())), "dtype_size is noexcept");
    static_assert(noexcept(dtype_name(std::declval<DType>())), "dtype_name is noexcept");
    static_assert(noexcept(is_real_float(std::declval<DType>())), "is_real_float is noexcept");
} // namespace SushiAI

namespace SushiAI::Eager
{
    static_assert(
        noexcept(find_boundaries(std::declval<const Graph::Graph&>(),
                std::declval<const Partition&>(),
                std::declval<SushiRuntime::span<const Graph::ValueID>>())),
        "find_boundaries is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(RegionEmitter(std::declval<SushiBLAS::Engine&>(),
                std::declval<const RegionProgram&>(), std::declval<Graph::TensorPool&>(),
                std::declval<const Graph::FusionSelector*>(), std::declval<RegionID>())),
        "RegionEmitter is noexcept");
    static_assert(
        noexcept(std::declval<RegionEmitter&>().emit(std::declval<SushiRuntime::Recorder&>())),
        "emit is noexcept");
    static_assert(
        noexcept(std::declval<const RegionEmitter&>().emit_count()),
        "emit_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionEmitter&>().operation_count()),
        "operation_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionEmitter&>().preludes()),
        "preludes is noexcept");
    static_assert(
        noexcept(std::declval<const RegionEmitter&>().selections()),
        "selections is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(RegionExecutor(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Graph::FusionSelector*>())),
        "RegionExecutor is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().compose_count()),
        "compose_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().region_count()),
        "region_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().emit_count(std::declval<RegionID>())),
        "emit_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().emit_count()),
        "emit_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().trailing_record_count()),
        "trailing_record_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().program_of(std::declval<RegionKey>())),
        "program_of is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().allocated_bytes()),
        "allocated_bytes is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().operation_count()),
        "operation_count is noexcept");
    static_assert(
        noexcept(std::declval<const RegionExecutor&>().last_report()),
        "last_report is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(std::declval<RegionNames&>().intern(std::declval<const RegionInfo&>())),
        "intern is noexcept");
    static_assert(
        noexcept(std::declval<const RegionNames&>().find(std::declval<std::string_view>(),
                std::declval<Direction>())),
        "find is noexcept");
    static_assert(
        noexcept(std::declval<const RegionNames&>().label_of(std::declval<RegionID>())),
        "label_of is noexcept");
    static_assert(noexcept(std::declval<const RegionNames&>().size()), "size is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(compile_region(std::declval<const Graph::Graph&>(),
                std::declval<const Partition&>(), std::declval<RegionKey>(),
                std::declval<const RegionBoundary&>(), std::declval<const ProgramOptions&>())),
        "compile_region is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(signature_of(std::declval<const Graph::Graph&>(), std::declval<const Partition&>(),
                std::declval<RegionKey>(),
                std::declval<SushiRuntime::span<const std::uint64_t>>())),
        "signature_of is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(RegionTable(std::declval<StorageIDSource&>())),
        "RegionTable is noexcept");
    static_assert(
        noexcept(std::declval<RegionTable&>().plan(std::declval<const Graph::Graph&>(),
                std::declval<const Partition&>(), std::declval<const BoundaryMap&>(),
                std::declval<SushiRuntime::span<const StorageID>>())),
        "plan is noexcept");
    static_assert(noexcept(std::declval<RegionTable&>().forget()), "forget is noexcept");
    static_assert(noexcept(std::declval<const RegionTable&>().names()), "names is noexcept");
    static_assert(
        noexcept(std::declval<const RegionTable&>().live_count()),
        "live_count is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(Scope(std::declval<ScopeTrace&>(), std::declval<std::string_view>())),
        "Scope is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(ScopeTrace(std::declval<const Graph::Graph&>())),
        "ScopeTrace is noexcept");
    static_assert(
        noexcept(std::declval<ScopeTrace&>().is_legal_scope_name(std::declval<std::string_view>())),
        "is_legal_scope_name is noexcept");
    static_assert(
        noexcept(std::declval<ScopeTrace&>().open(std::declval<std::string_view>())),
        "open is noexcept");
    static_assert(noexcept(std::declval<ScopeTrace&>().close()), "close is noexcept");
    static_assert(noexcept(std::declval<ScopeTrace&>().seal()), "seal is noexcept");
    static_assert(noexcept(std::declval<const ScopeTrace&>().depth()), "depth is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(Session(std::declval<SushiBLAS::Engine&>(), std::declval<StepExecutor&>(),
                std::declval<const RegionPartitioner&>(), std::declval<SessionOptions>())),
        "Session is noexcept");
    static_assert(
        noexcept(std::declval<Session&>().bind_state(std::declval<Graph::StateID>(),
                std::declval<const SushiBLAS::Tensor&>(), std::declval<bool>())),
        "bind_state is noexcept");
    static_assert(
        noexcept(std::declval<const Session&>().gradient(std::declval<Graph::StateID>())),
        "gradient is noexcept");
    static_assert(noexcept(std::declval<Session&>().begin_step()), "begin_step is noexcept");
    static_assert(noexcept(std::declval<const Session&>().step_count()), "step_count is noexcept");
    static_assert(noexcept(std::declval<const Session&>().last_cost()), "last_cost is noexcept");
    static_assert(noexcept(std::declval<const Session&>().last_plan()), "last_plan is noexcept");
    static_assert(
        noexcept(std::declval<const Session&>().region_names()),
        "region_names is noexcept");
    static_assert(noexcept(std::declval<const Session&>().storage()), "storage is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(noexcept(SlotLedger(std::declval<StorageIDSource&>())), "SlotLedger is noexcept");
    static_assert(noexcept(std::declval<SlotLedger&>().begin_step()), "begin_step is noexcept");
    static_assert(
        noexcept(std::declval<SlotLedger&>().keep(std::declval<RegionID>(),
                std::declval<ValuePosition>(),
                std::declval<SushiRuntime::span<const ReaderSlot>>())),
        "keep is noexcept");
    static_assert(
        noexcept(std::declval<SlotLedger&>().match_readers(std::declval<RegionID>(),
                std::declval<ValuePosition>(), std::declval<const Shape&>(), std::declval<DType>(),
                std::declval<SushiRuntime::span<const ReaderSlot>>(),
                std::declval<SushiRuntime::span<const ReaderSlot>>())),
        "match_readers is noexcept");
    static_assert(
        noexcept(std::declval<SlotLedger&>().place(std::declval<RegionID>(),
                std::declval<ValuePosition>(), std::declval<const Shape&>(), std::declval<DType>(),
                std::declval<SushiRuntime::span<const ReaderSlot>>())),
        "place is noexcept");
    static_assert(noexcept(std::declval<SlotLedger&>().end_step()), "end_step is noexcept");
    static_assert(noexcept(std::declval<SlotLedger&>().forget()), "forget is noexcept");
    static_assert(noexcept(std::declval<const SlotLedger&>().created()), "created is noexcept");
    static_assert(noexcept(std::declval<const SlotLedger&>().size()), "size is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(noexcept(std::declval<const Step&>().builder()), "builder is noexcept");
    static_assert(noexcept(std::declval<Step&>().scopes()), "scopes is noexcept");
    static_assert(noexcept(std::declval<const Step&>().graph()), "graph is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().state(std::declval<Graph::StateID>())),
        "state is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().input(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::string_view>())),
        "input is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().feed(std::declval<Graph::ValueID>(),
                std::declval<const SushiBLAS::Tensor&>())),
        "feed is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().keep(std::declval<const Graph::Traced&>())),
        "keep is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().backward(std::declval<const Graph::Traced&>())),
        "backward is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().apply(std::declval<Graph::StepRecorder&>())),
        "apply is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().read(std::declval<const Graph::Traced&>())),
        "read is noexcept");
    static_assert(
        noexcept(std::declval<Step&>().read_scalar(std::declval<const Graph::Traced&>())),
        "read_scalar is noexcept");
    static_assert(noexcept(std::declval<Step&>().finish()), "finish is noexcept");
    static_assert(noexcept(std::declval<const Step&>().has_run()), "has_run is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(
        noexcept(StorageBank(std::declval<SushiBLAS::Engine&>())),
        "StorageBank is noexcept");
    static_assert(
        noexcept(std::declval<StorageBank&>().adopt(std::declval<StorageID>(),
                std::declval<SushiBLAS::Tensor>())),
        "adopt is noexcept");
    static_assert(
        noexcept(std::declval<StorageBank&>().allocate(std::declval<StorageID>(),
                std::declval<const Shape&>(), std::declval<DType>())),
        "allocate is noexcept");
    static_assert(
        noexcept(std::declval<StorageBank&>().release(std::declval<StorageID>())),
        "release is noexcept");
    static_assert(
        noexcept(std::declval<const StorageBank&>().holds(std::declval<StorageID>())),
        "holds is noexcept");
    static_assert(
        noexcept(std::declval<StorageBank&>().tensor(std::declval<StorageID>())),
        "tensor is noexcept");
    static_assert(
        noexcept(std::declval<const StorageBank&>().tensor(std::declval<StorageID>())),
        "tensor is noexcept");
    static_assert(noexcept(std::declval<const StorageBank&>().size()), "size is noexcept");
    static_assert(
        noexcept(std::declval<const StorageBank&>().allocation_count()),
        "allocation_count is noexcept");
    static_assert(noexcept(std::declval<const StorageBank&>().engine()), "engine is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Eager
{
    static_assert(noexcept(std::declval<StorageIDSource&>().next()), "next is noexcept");
} // namespace SushiAI::Eager

namespace SushiAI::Test
{
    static_assert(noexcept(std::declval<TestEnvironment&>().runtime()), "runtime is noexcept");
} // namespace SushiAI::Test

namespace SushiAI::Test
{
    static_assert(noexcept(TestGraph(std::declval<SushiBLAS::Engine&>())), "TestGraph is noexcept");
    static_assert(noexcept(std::declval<TestGraph&>().recorder()), "recorder is noexcept");
    static_assert(
        noexcept(std::declval<TestGraph&>().graph_recorder()),
        "graph_recorder is noexcept");
    static_assert(noexcept(std::declval<TestGraph&>().run()), "run is noexcept");
} // namespace SushiAI::Test

namespace SushiAI
{
    static_assert(
        noexcept(set_fatal_handler(std::declval<FatalHandler>())),
        "set_fatal_handler is noexcept");
} // namespace SushiAI

namespace SushiAI
{
    static_assert(
        noexcept(status_code_name(std::declval<StatusCode>())),
        "status_code_name is noexcept");
    static_assert(noexcept(Status(std::declval<const Status&>())), "Status is noexcept");
    static_assert(noexcept(Status(std::declval<Status&&>())), "Status is noexcept");
    static_assert(noexcept(std::declval<const Status&>().is_ok()), "is_ok is noexcept");
    static_assert(noexcept(std::declval<const Status&>().code()), "code is noexcept");
    static_assert(noexcept(std::declval<const Status&>().message()), "message is noexcept");
} // namespace SushiAI

namespace SushiAI::Graph
{
    static_assert(
        noexcept(std::declval<ArenaSet&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<MemoryPlan>(), std::declval<const ArenaSet*>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const ArenaSet&>().plan()), "plan is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().holds(std::declval<ValueID>())),
        "holds is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().view(std::declval<ValueID>(),
                std::declval<const Shape&>(), std::declval<DType>())),
        "view is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().holds_scratch(std::declval<NodeID>(),
                std::declval<int>())),
        "holds_scratch is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().scratch_view(std::declval<NodeID>(),
                std::declval<int>(), std::declval<const Shape&>(), std::declval<DType>())),
        "scratch_view is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().arena_count()),
        "arena_count is noexcept");
    static_assert(noexcept(std::declval<const ArenaSet&>().bytes()), "bytes is noexcept");
    static_assert(
        noexcept(std::declval<const ArenaSet&>().owned_bytes()),
        "owned_bytes is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(output_extent(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>())),
        "output_extent is noexcept");
    static_assert(
        noexcept(window_extents(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<const ConvGeometry&>())),
        "window_extents is noexcept");
    static_assert(
        noexcept(upsampled_extents(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "upsampled_extents is noexcept");
    static_assert(
        noexcept(is_legal_leaky_relu_slope(std::declval<double>())),
        "is_legal_leaky_relu_slope is noexcept");
    static_assert(
        noexcept(dropout_seed(std::declval<const Attributes&>())),
        "dropout_seed is noexcept");
    static_assert(noexcept(GraphBuilder(std::declval<Graph&>())), "GraphBuilder is noexcept");
    static_assert(noexcept(std::declval<const GraphBuilder&>().graph()), "graph is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().input(std::declval<Shape>(),
                std::declval<std::string_view>(), std::declval<DType>())),
        "input is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().parameter(std::declval<Shape>(),
                std::declval<std::string_view>(), std::declval<DType>())),
        "parameter is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().buffer(std::declval<Shape>(),
                std::declval<std::string_view>(), std::declval<DType>())),
        "buffer is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().matmul(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<bool>(), std::declval<bool>())),
        "matmul is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().bias_add(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "bias_add is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().sum_rows(std::declval<const Traced&>())),
        "sum_rows is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().gather_rows(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "gather_rows is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().scatter_add_rows(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<int64_t>())),
        "scatter_add_rows is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().causal_mask(std::declval<const Traced&>(),
                std::declval<double>())),
        "causal_mask is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().add(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().sub(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "sub is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().mul(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "mul is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().div(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "div is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().minimum(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "minimum is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().maximum(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "maximum is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().less_equal(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "less_equal is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().select(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>())),
        "select is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().scale(std::declval<const Traced&>(),
                std::declval<double>())),
        "scale is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().dynamic_scale(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<std::string_view>())),
        "dynamic_scale is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().exp(std::declval<const Traced&>())),
        "exp is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().abs(std::declval<const Traced&>())),
        "abs is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().atan(std::declval<const Traced&>())),
        "atan is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().softplus(std::declval<const Traced&>())),
        "softplus is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().zeros(std::declval<Shape>(),
                std::declval<DType>())),
        "zeros is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().zeros_like(std::declval<const Traced&>(),
                std::declval<Shape>(), std::declval<DType>())),
        "zeros_like is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().constant(std::declval<Shape>(),
                std::declval<double>(), std::declval<DType>())),
        "constant is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().constant_like(std::declval<const Traced&>(),
                std::declval<double>())),
        "constant_like is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().relu(std::declval<const Traced&>())),
        "relu is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().gelu(std::declval<const Traced&>())),
        "gelu is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().sigmoid(std::declval<const Traced&>())),
        "sigmoid is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().tanh(std::declval<const Traced&>())),
        "tanh is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().silu(std::declval<const Traced&>())),
        "silu is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().leaky_relu(std::declval<const Traced&>(),
                std::declval<double>())),
        "leaky_relu is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().activation_backward(std::declval<OpID>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Attributes&>())),
        "activation_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().log_softmax(std::declval<const Traced&>())),
        "log_softmax is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().softmax(std::declval<const Traced&>())),
        "softmax is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().log_softmax_backward(
                std::declval<const Traced&>(), std::declval<const Traced&>())),
        "log_softmax_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().nll_loss(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "nll_loss is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().accumulate(std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "accumulate is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().layout_convert(std::declval<const Traced&>(),
                std::declval<TensorLayout>())),
        "layout_convert is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().cast(std::declval<const Traced&>(),
                std::declval<DType>(), std::declval<std::string_view>())),
        "cast is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().cast_into(std::declval<const Traced&>(),
                std::declval<ValueID>())),
        "cast_into is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().like(std::declval<ValueID>(),
                std::declval<std::string_view>())),
        "like is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().declare_like(std::declval<ValueID>(),
                std::declval<Shape>(), std::declval<DType>(), std::declval<Lifetime>(),
                std::declval<std::string_view>())),
        "declare_like is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().im2col(std::declval<const Traced&>(),
                std::declval<const ConvGeometry&>())),
        "im2col is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().im2col_with_extents(
                std::declval<const Traced&>(), std::declval<const ConvGeometry&>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "im2col_with_extents is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().col2im(std::declval<const Traced&>(),
                std::declval<Shape>(), std::declval<const ConvGeometry&>())),
        "col2im is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().col2im_with_extents(
                std::declval<const Traced&>(), std::declval<Shape>(),
                std::declval<const ConvGeometry&>(), std::declval<int64_t>(), std::declval<int64_t>(
                ))),
        "col2im_with_extents is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().conv2d(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const ConvGeometry&>())),
        "conv2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().depthwise_conv2d(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const ConvGeometry&>())),
        "depthwise_conv2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().depthwise_conv2d_backward_input(
                std::declval<const Traced&>(), std::declval<const Traced&>(), std::declval<Shape>(),
                std::declval<const ConvGeometry&>())),
        "depthwise_conv2d_backward_input is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().depthwise_conv2d_backward_weight(
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const ConvGeometry&>())),
        "depthwise_conv2d_backward_weight is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().maxpool2d(std::declval<const Traced&>(),
                std::declval<const ConvGeometry&>())),
        "maxpool2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().maxpool2d_with_extents(
                std::declval<const Traced&>(), std::declval<const ConvGeometry&>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "maxpool2d_with_extents is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().avgpool2d(std::declval<const Traced&>(),
                std::declval<const ConvGeometry&>())),
        "avgpool2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().upsample2d(std::declval<const Traced&>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "upsample2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().spatial_mean(std::declval<const Traced&>())),
        "spatial_mean is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().maxpool2d_backward(std::declval<const Traced&>(
                ), std::declval<const Traced&>(), std::declval<Shape>(),
                std::declval<const ConvGeometry&>())),
        "maxpool2d_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().avgpool2d_backward(std::declval<const Traced&>(
                ), std::declval<Shape>(), std::declval<const ConvGeometry&>())),
        "avgpool2d_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().upsample2d_backward(
                std::declval<const Traced&>(), std::declval<Shape>(), std::declval<int64_t>(),
                std::declval<int64_t>())),
        "upsample2d_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().spatial_mean_backward(
                std::declval<const Traced&>(), std::declval<Shape>())),
        "spatial_mean_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().concat_channels(
                std::declval<SushiRuntime::span<const ValueID>>())),
        "concat_channels is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().slice_channels(std::declval<const Traced&>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "slice_channels is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().channel_affine(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>())),
        "channel_affine is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().sum_channels(std::declval<const Traced&>())),
        "sum_channels is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().batchnorm2d(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<double>())),
        "batchnorm2d is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().batchnorm2d_backward(
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "batchnorm2d_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().batchnorm_fold(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<double>())),
        "batchnorm_fold is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().layer_norm(std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<double>())),
        "layer_norm is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().layer_norm_backward(
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Traced&>(), std::declval<const Traced&>(),
                std::declval<const Traced&>())),
        "layer_norm_backward is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().dropout(std::declval<const Traced&>(),
                std::declval<double>(), std::declval<std::uint64_t>())),
        "dropout is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().reshape(std::declval<const Traced&>(),
                std::declval<Shape>())),
        "reshape is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().stop_gradient(std::declval<const Traced&>())),
        "stop_gradient is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().refuse(std::declval<Status>())),
        "refuse is noexcept");
    static_assert(
        noexcept(std::declval<const GraphBuilder&>().observe(std::declval<Status>())),
        "observe is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(std::declval<const CastReport&>().casts()), "casts is noexcept");
    static_assert(noexcept(std::declval<const CastReport&>().count()), "count is noexcept");
    static_assert(
        noexcept(std::declval<const CastReport&>().count_into(std::declval<DType>())),
        "count_into is noexcept");
    static_assert(
        noexcept(std::declval<const CastReport&>().nodes_before()),
        "nodes_before is noexcept");
    static_assert(
        noexcept(std::declval<const CastReport&>().nodes_after()),
        "nodes_after is noexcept");
    static_assert(
        noexcept(std::declval<const CastReport&>().retyped_values()),
        "retyped_values is noexcept");
    static_assert(noexcept(std::declval<const CastReport&>().policy()), "policy is noexcept");
    static_assert(noexcept(std::declval<const CastReport&>().to_string()), "to_string is noexcept");
    static_assert(
        noexcept(insert_casts(std::declval<Graph&>(), std::declval<const CastOptions&>())),
        "insert_casts is noexcept");
    static_assert(
        noexcept(find_dtype_mismatches(std::declval<const Graph&>(),
                std::declval<const PrecisionPolicy&>())),
        "find_dtype_mismatches is noexcept");
    static_assert(noexcept(describe(std::declval<const DTypeMismatch&>())), "describe is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(std::declval<CompiledStep&>().compile(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Graph&>(), std::declval<TensorPool&>(),
                std::declval<StepRecorder*>(), std::declval<const FusionSelector*>(),
                std::declval<NodeRange>(), std::declval<std::string_view>())),
        "compile is noexcept");
    static_assert(noexcept(std::declval<CompiledStep&>().run()), "run is noexcept");
    static_assert(
        noexcept(std::declval<CompiledStep&>().run_at(std::declval<std::uint64_t>())),
        "run_at is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().replay_prelude_count()),
        "replay_prelude_count is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().has_trailing_work()),
        "has_trailing_work is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().compile_count()),
        "compile_count is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().run_count()),
        "run_count is noexcept");
    static_assert(
        noexcept(std::declval<CompiledStep&>().restore_run_count(std::declval<std::size_t>())),
        "restore_run_count is noexcept");
    static_assert(noexcept(std::declval<const CompiledStep&>().empty()), "empty is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().operation_count()),
        "operation_count is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().selections()),
        "selections is noexcept");
    static_assert(
        noexcept(std::declval<const CompiledStep&>().last_report()),
        "last_report is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(splitmix64_at(std::declval<std::uint64_t>(), std::declval<std::uint64_t>())),
        "splitmix64_at is noexcept");
    static_assert(
        noexcept(dropout_site_seed(std::declval<std::uint64_t>(), std::declval<std::uint64_t>())),
        "dropout_site_seed is noexcept");
    static_assert(
        noexcept(dropout_bits(std::declval<std::uint64_t>(), std::declval<std::uint64_t>(),
                std::declval<std::uint64_t>())),
        "dropout_bits is noexcept");
    static_assert(
        noexcept(dropout_keeps(std::declval<std::uint64_t>(), std::declval<std::uint64_t>(),
                std::declval<std::uint64_t>(), std::declval<double>())),
        "dropout_keeps is noexcept");
    static_assert(
        noexcept(fill_keep_mask(std::declval<float*>(), std::declval<std::size_t>(),
                std::declval<std::uint64_t>(), std::declval<std::uint64_t>(),
                std::declval<double>())),
        "fill_keep_mask is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(fused_form_name(std::declval<FusedForm>())),
        "fused_form_name is noexcept");
    static_assert(
        noexcept(fused_form_from_name(std::declval<std::string_view>())),
        "fused_form_from_name is noexcept");
    static_assert(noexcept(fused_form_op(std::declval<FusedForm>())), "fused_form_op is noexcept");
    static_assert(noexcept(fused_form_of(std::declval<OpID>())), "fused_form_of is noexcept");
    static_assert(
        noexcept(is_gemm_epilogue(std::declval<FusedForm>())),
        "is_gemm_epilogue is noexcept");
    static_assert(
        noexcept(fused_form_activation(std::declval<FusedForm>())),
        "fused_form_activation is noexcept");
    static_assert(
        noexcept(fused_form_spills(std::declval<FusedForm>())),
        "fused_form_spills is noexcept");
    static_assert(
        noexcept(catalog_folds_activation(std::declval<OpID>())),
        "catalog_folds_activation is noexcept");
    static_assert(
        noexcept(fusion_decline_reason(std::declval<FusionDecline>())),
        "fusion_decline_reason is noexcept");
    static_assert(noexcept(std::declval<const FusionReport&>().fused()), "fused is noexcept");
    static_assert(noexcept(std::declval<const FusionReport&>().declined()), "declined is noexcept");
    static_assert(
        noexcept(std::declval<const FusionReport&>().nodes_before()),
        "nodes_before is noexcept");
    static_assert(
        noexcept(std::declval<const FusionReport&>().nodes_after()),
        "nodes_after is noexcept");
    static_assert(noexcept(std::declval<const FusionReport&>().enabled()), "enabled is noexcept");
    static_assert(
        noexcept(std::declval<const FusionReport&>().count(std::declval<FusedForm>())),
        "count is noexcept");
    static_assert(
        noexcept(std::declval<const FusionReport&>().to_string()),
        "to_string is noexcept");
    static_assert(
        noexcept(fuse(std::declval<Graph&>(), std::declval<const FusionOptions&>())),
        "fuse is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(DeviceFusionSelector(std::declval<const SushiBLAS::Device::Profile&>(),
                std::declval<SushiBLAS::ExecutionMode>())),
        "DeviceFusionSelector is noexcept");
    static_assert(
        noexcept(std::declval<DeviceFusionSelector&>().for_engine(std::declval<SushiBLAS::Engine&>(
                ))),
        "for_engine is noexcept");
    static_assert(
        noexcept(std::declval<const DeviceFusionSelector&>().supports_fp64()),
        "supports_fp64 is noexcept");
    static_assert(noexcept(std::declval<const DeviceFusionSelector&>().mode()), "mode is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(lifetime_name(std::declval<Lifetime>())), "lifetime_name is noexcept");
    static_assert(noexcept(is_persistent(std::declval<Lifetime>())), "is_persistent is noexcept");
    static_assert(
        noexcept(std::declval<const Attributes&>().get(std::declval<std::size_t>())),
        "get is noexcept");
    static_assert(
        noexcept(std::declval<const Attributes&>().flag(std::declval<std::size_t>())),
        "flag is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().add_value(std::declval<Shape>(), std::declval<DType>(),
                std::declval<Lifetime>(), std::declval<std::string_view>())),
        "add_value is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().add_node(std::declval<OpID>(),
                std::declval<std::vector<ValueID>>(), std::declval<std::vector<ValueID>>(),
                std::declval<Attributes>())),
        "add_node is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().insert_node(std::declval<NodeID>(), std::declval<Node>())),
        "insert_node is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().value(std::declval<ValueID>())),
        "value is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().value(std::declval<ValueID>())),
        "value is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().node(std::declval<NodeID>())),
        "node is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().node(std::declval<NodeID>())),
        "node is noexcept");
    static_assert(noexcept(std::declval<const Graph&>().value_count()), "value_count is noexcept");
    static_assert(noexcept(std::declval<const Graph&>().node_count()), "node_count is noexcept");
    static_assert(noexcept(std::declval<const Graph&>().values()), "values is noexcept");
    static_assert(noexcept(std::declval<const Graph&>().nodes()), "nodes is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().state_leaf(std::declval<StateID>())),
        "state_leaf is noexcept");
    static_assert(
        noexcept(std::declval<Graph&>().bind_state_leaf(std::declval<StateID>(),
                std::declval<ValueID>())),
        "bind_state_leaf is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().state_leaves()),
        "state_leaves is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().without_nodes()),
        "without_nodes is noexcept");
    static_assert(noexcept(std::declval<Graph&>().mark_refused()), "mark_refused is noexcept");
    static_assert(noexcept(std::declval<const Graph&>().has_refusal()), "has_refusal is noexcept");
    static_assert(noexcept(std::declval<Graph&>().finalise()), "finalise is noexcept");
    static_assert(
        noexcept(std::declval<const Graph&>().prove_layout_uniformity()),
        "prove_layout_uniformity is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(scratch_of(std::declval<const Graph&>())), "scratch_of is noexcept");
    static_assert(
        noexcept(Lowering(std::declval<SushiBLAS::TaskRecorder&>(),
                std::declval<SushiBLAS::RNGStream&>(), std::declval<TensorPool&>(),
                std::declval<const FusionSelector&>())),
        "Lowering is noexcept");
    static_assert(
        noexcept(std::declval<Lowering&>().set_origin_prefix(std::declval<std::string>())),
        "set_origin_prefix is noexcept");
    static_assert(
        noexcept(std::declval<Lowering&>().record(std::declval<const Graph&>(),
                std::declval<NodeRange>())),
        "record is noexcept");
    static_assert(
        noexcept(std::declval<const Lowering&>().emitted_operations()),
        "emitted_operations is noexcept");
    static_assert(noexcept(std::declval<const Lowering&>().selections()), "selections is noexcept");
    static_assert(noexcept(std::declval<Lowering&>().take_preludes()), "take_preludes is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_relu(std::declval<const LoweringContext&>())),
        "lower_relu is noexcept");
    static_assert(
        noexcept(lower_gelu(std::declval<const LoweringContext&>())),
        "lower_gelu is noexcept");
    static_assert(
        noexcept(lower_sigmoid(std::declval<const LoweringContext&>())),
        "lower_sigmoid is noexcept");
    static_assert(
        noexcept(lower_tanh(std::declval<const LoweringContext&>())),
        "lower_tanh is noexcept");
    static_assert(
        noexcept(lower_silu(std::declval<const LoweringContext&>())),
        "lower_silu is noexcept");
    static_assert(
        noexcept(lower_softplus(std::declval<const LoweringContext&>())),
        "lower_softplus is noexcept");
    static_assert(
        noexcept(lower_leaky_relu(std::declval<const LoweringContext&>())),
        "lower_leaky_relu is noexcept");
    static_assert(
        noexcept(lower_relu_backward(std::declval<const LoweringContext&>())),
        "lower_relu_backward is noexcept");
    static_assert(
        noexcept(lower_gelu_backward(std::declval<const LoweringContext&>())),
        "lower_gelu_backward is noexcept");
    static_assert(
        noexcept(lower_sigmoid_backward(std::declval<const LoweringContext&>())),
        "lower_sigmoid_backward is noexcept");
    static_assert(
        noexcept(lower_tanh_backward(std::declval<const LoweringContext&>())),
        "lower_tanh_backward is noexcept");
    static_assert(
        noexcept(lower_silu_backward(std::declval<const LoweringContext&>())),
        "lower_silu_backward is noexcept");
    static_assert(
        noexcept(lower_softplus_backward(std::declval<const LoweringContext&>())),
        "lower_softplus_backward is noexcept");
    static_assert(
        noexcept(lower_leaky_relu_backward(std::declval<const LoweringContext&>())),
        "lower_leaky_relu_backward is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(find_lowering(std::declval<OpID>())), "find_lowering is noexcept");
    static_assert(
        noexcept(find_scratch_rule(std::declval<OpID>())),
        "find_scratch_rule is noexcept");
    static_assert(noexcept(is_unlowered(std::declval<OpID>())), "is_unlowered is noexcept");
    static_assert(
        noexcept(lowering_row_count(std::declval<OpID>())),
        "lowering_row_count is noexcept");
    static_assert(
        noexcept(every_identity_is_accounted_for(
                std::declval<SushiRuntime::span<const Ops::OpName>>())),
        "every_identity_is_accounted_for is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(transpose_of(std::declval<bool>())), "transpose_of is noexcept");
    static_assert(
        noexcept(LoweringContext(std::declval<SushiBLAS::TaskRecorder&>(),
                std::declval<SushiBLAS::RNGStream&>(), std::declval<TensorPool&>(),
                std::declval<const Graph&>(), std::declval<NodeID>(),
                std::declval<const FusionSelector&>(),
                std::declval<std::vector<FusionSelection>&>(), std::declval<std::size_t&>())),
        "LoweringContext is noexcept");
    static_assert(
        noexcept(LoweringContext(std::declval<SushiBLAS::TaskRecorder&>(),
                std::declval<SushiBLAS::RNGStream&>(), std::declval<TensorPool&>(),
                std::declval<const Graph&>(), std::declval<NodeID>(),
                std::declval<const FusionSelector&>(),
                std::declval<std::vector<FusionSelection>&>(), std::declval<std::size_t&>(),
                std::declval<std::vector<ReplayPrelude>&>())),
        "LoweringContext is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().recorder()),
        "recorder is noexcept");
    static_assert(noexcept(std::declval<const LoweringContext&>().stream()), "stream is noexcept");
    static_assert(noexcept(std::declval<const LoweringContext&>().graph()), "graph is noexcept");
    static_assert(noexcept(std::declval<const LoweringContext&>().id()), "id is noexcept");
    static_assert(noexcept(std::declval<const LoweringContext&>().node()), "node is noexcept");
    static_assert(noexcept(std::declval<const LoweringContext&>().name()), "name is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().require_arity(std::declval<std::size_t>(),
                std::declval<std::size_t>())),
        "require_arity is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().require_arity_at_least(
                std::declval<std::size_t>(), std::declval<std::size_t>())),
        "require_arity_at_least is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().require_variadic_arity(
                std::declval<std::size_t>(), std::declval<std::size_t>())),
        "require_variadic_arity is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().input_value(std::declval<std::size_t>())),
        "input_value is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().output_value(std::declval<std::size_t>())),
        "output_value is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().input(std::declval<std::size_t>())),
        "input is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().mutable_input(std::declval<std::size_t>())),
        "mutable_input is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().output(std::declval<std::size_t>())),
        "output is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().output_is_bound(std::declval<std::size_t>(
                ))),
        "output_is_bound is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().alias_output(std::declval<std::size_t>(),
                std::declval<std::size_t>())),
        "alias_output is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().scratch(std::declval<int>(),
                std::declval<const Shape&>(), std::declval<DType>())),
        "scratch is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().ones(std::declval<const Shape&>(),
                std::declval<DType>())),
        "ones is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().count(std::declval<std::size_t>())),
        "count is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().emitted()),
        "emitted is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().select(std::declval<const FusionQuery&>())),
        "select is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().close_selection(std::declval<std::size_t>(
                ))),
        "close_selection is noexcept");
    static_assert(
        noexcept(std::declval<const LoweringContext&>().before_each_replay(
                std::declval<ReplayPrelude>())),
        "before_each_replay is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_add(std::declval<const LoweringContext&>())),
        "lower_add is noexcept");
    static_assert(
        noexcept(lower_sub(std::declval<const LoweringContext&>())),
        "lower_sub is noexcept");
    static_assert(
        noexcept(lower_mul(std::declval<const LoweringContext&>())),
        "lower_mul is noexcept");
    static_assert(
        noexcept(lower_div(std::declval<const LoweringContext&>())),
        "lower_div is noexcept");
    static_assert(
        noexcept(lower_minimum(std::declval<const LoweringContext&>())),
        "lower_minimum is noexcept");
    static_assert(
        noexcept(lower_maximum(std::declval<const LoweringContext&>())),
        "lower_maximum is noexcept");
    static_assert(
        noexcept(lower_scale(std::declval<const LoweringContext&>())),
        "lower_scale is noexcept");
    static_assert(
        noexcept(lower_dynamic_scale(std::declval<const LoweringContext&>())),
        "lower_dynamic_scale is noexcept");
    static_assert(
        noexcept(lower_exp(std::declval<const LoweringContext&>())),
        "lower_exp is noexcept");
    static_assert(
        noexcept(lower_abs(std::declval<const LoweringContext&>())),
        "lower_abs is noexcept");
    static_assert(
        noexcept(lower_atan(std::declval<const LoweringContext&>())),
        "lower_atan is noexcept");
    static_assert(
        noexcept(lower_less_equal(std::declval<const LoweringContext&>())),
        "lower_less_equal is noexcept");
    static_assert(
        noexcept(lower_select(std::declval<const LoweringContext&>())),
        "lower_select is noexcept");
    static_assert(
        noexcept(lower_zero(std::declval<const LoweringContext&>())),
        "lower_zero is noexcept");
    static_assert(
        noexcept(lower_constant(std::declval<const LoweringContext&>())),
        "lower_constant is noexcept");
    static_assert(
        noexcept(lower_accumulate(std::declval<const LoweringContext&>())),
        "lower_accumulate is noexcept");
    static_assert(
        noexcept(lower_cast(std::declval<const LoweringContext&>())),
        "lower_cast is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_gemm_bias(std::declval<const LoweringContext&>())),
        "lower_gemm_bias is noexcept");
    static_assert(
        noexcept(lower_gemm_bias_relu(std::declval<const LoweringContext&>())),
        "lower_gemm_bias_relu is noexcept");
    static_assert(
        noexcept(lower_gemm_bias_sigmoid(std::declval<const LoweringContext&>())),
        "lower_gemm_bias_sigmoid is noexcept");
    static_assert(
        noexcept(lower_gemm_bias_tanh(std::declval<const LoweringContext&>())),
        "lower_gemm_bias_tanh is noexcept");
    static_assert(
        noexcept(lower_gemm_bias_gelu(std::declval<const LoweringContext&>())),
        "lower_gemm_bias_gelu is noexcept");
    static_assert(
        noexcept(lower_exp_sub_scale(std::declval<const LoweringContext&>())),
        "lower_exp_sub_scale is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_matmul(std::declval<const LoweringContext&>())),
        "lower_matmul is noexcept");
    static_assert(
        noexcept(lower_bias_add(std::declval<const LoweringContext&>())),
        "lower_bias_add is noexcept");
    static_assert(
        noexcept(lower_sum_rows(std::declval<const LoweringContext&>())),
        "lower_sum_rows is noexcept");
    static_assert(
        noexcept(lower_gather_rows(std::declval<const LoweringContext&>())),
        "lower_gather_rows is noexcept");
    static_assert(
        noexcept(lower_scatter_add_rows(std::declval<const LoweringContext&>())),
        "lower_scatter_add_rows is noexcept");
    static_assert(
        noexcept(lower_causal_mask(std::declval<const LoweringContext&>())),
        "lower_causal_mask is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_log_softmax(std::declval<const LoweringContext&>())),
        "lower_log_softmax is noexcept");
    static_assert(
        noexcept(lower_log_softmax_backward(std::declval<const LoweringContext&>())),
        "lower_log_softmax_backward is noexcept");
    static_assert(
        noexcept(lower_nll_loss(std::declval<const LoweringContext&>())),
        "lower_nll_loss is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_batchnorm2d(std::declval<const LoweringContext&>())),
        "lower_batchnorm2d is noexcept");
    static_assert(
        noexcept(lower_batchnorm2d_backward(std::declval<const LoweringContext&>())),
        "lower_batchnorm2d_backward is noexcept");
    static_assert(
        noexcept(request_batchnorm2d_scratch(std::declval<const Graph&>(), std::declval<NodeID>(),
                std::declval<std::vector<ScratchRequest>&>())),
        "request_batchnorm2d_scratch is noexcept");
    static_assert(
        noexcept(request_batchnorm2d_backward_scratch(std::declval<const Graph&>(),
                std::declval<NodeID>(), std::declval<std::vector<ScratchRequest>&>())),
        "request_batchnorm2d_backward_scratch is noexcept");
    static_assert(
        noexcept(lower_channel_affine(std::declval<const LoweringContext&>())),
        "lower_channel_affine is noexcept");
    static_assert(
        noexcept(lower_batchnorm_fold(std::declval<const LoweringContext&>())),
        "lower_batchnorm_fold is noexcept");
    static_assert(
        noexcept(lower_layer_norm(std::declval<const LoweringContext&>())),
        "lower_layer_norm is noexcept");
    static_assert(
        noexcept(lower_layer_norm_backward(std::declval<const LoweringContext&>())),
        "lower_layer_norm_backward is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_dropout(std::declval<const LoweringContext&>())),
        "lower_dropout is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_leaf(std::declval<const LoweringContext&>())),
        "lower_leaf is noexcept");
    static_assert(
        noexcept(lower_reshape(std::declval<const LoweringContext&>())),
        "lower_reshape is noexcept");
    static_assert(
        noexcept(lower_layout_convert(std::declval<const LoweringContext&>())),
        "lower_layout_convert is noexcept");
    static_assert(
        noexcept(lower_stop_gradient(std::declval<const LoweringContext&>())),
        "lower_stop_gradient is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(lower_im2col(std::declval<const LoweringContext&>())),
        "lower_im2col is noexcept");
    static_assert(
        noexcept(lower_col2im(std::declval<const LoweringContext&>())),
        "lower_col2im is noexcept");
    static_assert(
        noexcept(lower_maxpool2d(std::declval<const LoweringContext&>())),
        "lower_maxpool2d is noexcept");
    static_assert(
        noexcept(lower_maxpool2d_backward(std::declval<const LoweringContext&>())),
        "lower_maxpool2d_backward is noexcept");
    static_assert(
        noexcept(lower_avgpool2d(std::declval<const LoweringContext&>())),
        "lower_avgpool2d is noexcept");
    static_assert(
        noexcept(lower_avgpool2d_backward(std::declval<const LoweringContext&>())),
        "lower_avgpool2d_backward is noexcept");
    static_assert(
        noexcept(lower_upsample2d(std::declval<const LoweringContext&>())),
        "lower_upsample2d is noexcept");
    static_assert(
        noexcept(lower_upsample2d_backward(std::declval<const LoweringContext&>())),
        "lower_upsample2d_backward is noexcept");
    static_assert(
        noexcept(lower_spatial_mean(std::declval<const LoweringContext&>())),
        "lower_spatial_mean is noexcept");
    static_assert(
        noexcept(lower_spatial_mean_backward(std::declval<const LoweringContext&>())),
        "lower_spatial_mean_backward is noexcept");
    static_assert(
        noexcept(lower_concat_channels(std::declval<const LoweringContext&>())),
        "lower_concat_channels is noexcept");
    static_assert(
        noexcept(lower_slice_channels(std::declval<const LoweringContext&>())),
        "lower_slice_channels is noexcept");
    static_assert(
        noexcept(lower_depthwise_conv2d(std::declval<const LoweringContext&>())),
        "lower_depthwise_conv2d is noexcept");
    static_assert(
        noexcept(lower_depthwise_conv2d_backward_input(std::declval<const LoweringContext&>())),
        "lower_depthwise_conv2d_backward_input is noexcept");
    static_assert(
        noexcept(lower_depthwise_conv2d_backward_weight(std::declval<const LoweringContext&>())),
        "lower_depthwise_conv2d_backward_weight is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(std::declval<MemoryPlan&>().build(std::declval<const Graph&>(),
                std::declval<const PlanOptions&>(),
                std::declval<SushiRuntime::span<const ScratchRequest>>())),
        "build is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().value_count()),
        "value_count is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().placement(std::declval<ValueID>())),
        "placement is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().is_packed(std::declval<ValueID>())),
        "is_packed is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().buffer_of(std::declval<ValueID>())),
        "buffer_of is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().scratch_placement(std::declval<NodeID>(),
                std::declval<int>())),
        "scratch_placement is noexcept");
    static_assert(noexcept(std::declval<const MemoryPlan&>().arenas()), "arenas is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().arena_bytes()),
        "arena_bytes is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().unpacked_bytes()),
        "unpacked_bytes is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().reuse_enabled()),
        "reuse_enabled is noexcept");
    static_assert(
        noexcept(std::declval<const MemoryPlan&>().reuse_note()),
        "reuse_note is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(MixedPrecisionPolicy()), "MixedPrecisionPolicy is noexcept");
    static_assert(
        noexcept(std::declval<MixedPrecisionPolicy&>().make(std::declval<DType>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const MixedPrecisionPolicy&>().low()), "low is noexcept");
    static_assert(
        noexcept(narrows_under_mixed_precision(std::declval<OpID>())),
        "narrows_under_mixed_precision is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(rematerialise_im2col(std::declval<Graph&>())),
        "rematerialise_im2col is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(StateIdentity(std::declval<const StateIdentity&>())),
        "StateIdentity is noexcept");
    static_assert(
        noexcept(StateIdentity(std::declval<StateIdentity&&>())),
        "StateIdentity is noexcept");
    static_assert(noexcept(std::declval<StateIdentity&>().get()), "get is noexcept");
    static_assert(noexcept(std::declval<const StateIdentity&>().peek()), "peek is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(std::declval<StepRecorderList&>().add(std::declval<StepRecorder&>())),
        "add is noexcept");
    static_assert(noexcept(std::declval<const StepRecorderList&>().empty()), "empty is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(
        noexcept(StepRecording(std::declval<SushiBLAS::TaskRecorder&>(),
                std::declval<SushiRuntime::Recorder&>())),
        "StepRecording is noexcept");
    static_assert(noexcept(std::declval<const StepRecording&>().tasks()), "tasks is noexcept");
    static_assert(noexcept(std::declval<const StepRecording&>().graph()), "graph is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(std::declval<const PoolBytes&>().total()), "total is noexcept");
    static_assert(
        noexcept(TensorPool(std::declval<SushiBLAS::Engine&>())),
        "TensorPool is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().adopt(std::declval<MemoryPlan>(),
                std::declval<const TensorPool*>())),
        "adopt is noexcept");
    static_assert(noexcept(std::declval<const TensorPool&>().has_plan()), "has_plan is noexcept");
    static_assert(noexcept(std::declval<const TensorPool&>().arenas()), "arenas is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().create(std::declval<const Shape&>(),
                std::declval<DType>())),
        "create is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().bind(std::declval<ValueID>(),
                std::declval<SushiBLAS::Tensor>())),
        "bind is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().is_bound(std::declval<ValueID>())),
        "is_bound is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().tensor(std::declval<ValueID>())),
        "tensor is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().tensor(std::declval<ValueID>())),
        "tensor is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().materialise(std::declval<ValueID>(),
                std::declval<const Value&>())),
        "materialise is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().materialise_leaves(std::declval<const Graph&>())),
        "materialise_leaves is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().alias(std::declval<ValueID>(),
                std::declval<ValueID>())),
        "alias is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().ones(std::declval<const Shape&>(),
                std::declval<DType>())),
        "ones is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().scratch(std::declval<NodeID>(), std::declval<int>(),
                std::declval<const Shape&>(), std::declval<DType>())),
        "scratch is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().request_constant(std::declval<SushiBLAS::Tensor&>(),
                std::declval<double>())),
        "request_constant is noexcept");
    static_assert(
        noexcept(std::declval<TensorPool&>().flush_constants()),
        "flush_constants is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().tensor_count()),
        "tensor_count is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().allocated_bytes()),
        "allocated_bytes is noexcept");
    static_assert(noexcept(std::declval<const TensorPool&>().bytes()), "bytes is noexcept");
    static_assert(
        noexcept(std::declval<const TensorPool&>().unpacked_bytes()),
        "unpacked_bytes is noexcept");
    static_assert(noexcept(std::declval<const TensorPool&>().engine()), "engine is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Graph
{
    static_assert(noexcept(Traced(std::declval<ValueID>())), "Traced is noexcept");
    static_assert(noexcept(Traced(std::declval<Status>())), "Traced is noexcept");
    static_assert(noexcept(std::declval<const Traced&>().has_value()), "has_value is noexcept");
    static_assert(noexcept(std::declval<const Traced&>().status()), "status is noexcept");
    static_assert(noexcept(std::declval<const Traced&>().value()), "value is noexcept");
} // namespace SushiAI::Graph

namespace SushiAI::Hash
{
    static_assert(
        noexcept(fnv1a_byte(std::declval<std::uint64_t>(), std::declval<unsigned char>())),
        "fnv1a_byte is noexcept");
    static_assert(
        noexcept(fnv1a_text(std::declval<std::uint64_t>(), std::declval<std::string_view>())),
        "fnv1a_text is noexcept");
    static_assert(
        noexcept(fnv1a_word(std::declval<std::uint64_t>(), std::declval<std::uint64_t>())),
        "fnv1a_word is noexcept");
} // namespace SushiAI::Hash

namespace SushiAI::Ops
{
    static_assert(noexcept(op(std::declval<std::string_view>())), "op is noexcept");
    static_assert(
        noexcept(differentiable_op(std::declval<std::string_view>())),
        "differentiable_op is noexcept");
    static_assert(
        noexcept(gradient_barrier_op(std::declval<std::string_view>())),
        "gradient_barrier_op is noexcept");
    static_assert(
        noexcept(all_ids_unique(std::declval<SushiRuntime::span<const OpName>>())),
        "all_ids_unique is noexcept");
    static_assert(noexcept(name_of(std::declval<OpID>())), "name_of is noexcept");
    static_assert(
        noexcept(any_barrier_is_differentiable(std::declval<SushiRuntime::span<const OpName>>())),
        "any_barrier_is_differentiable is noexcept");
    static_assert(
        noexcept(is_gradient_barrier(std::declval<OpID>())),
        "is_gradient_barrier is noexcept");
    static_assert(noexcept(activation_of(std::declval<OpID>())), "activation_of is noexcept");
    static_assert(
        noexcept(is_activation_backward(std::declval<OpID>())),
        "is_activation_backward is noexcept");
    static_assert(
        noexcept(activation_survives_fusion(std::declval<OpID>())),
        "activation_survives_fusion is noexcept");
} // namespace SushiAI::Ops

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<AdamW&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::vector<ParameterSlot>>(), std::declval<AdamWOptions>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const AdamW&>().options()), "options is noexcept");
    static_assert(noexcept(std::declval<AdamW&>().options()), "options is noexcept");
    static_assert(
        noexcept(std::declval<const AdamW&>().first_moment(std::declval<std::size_t>())),
        "first_moment is noexcept");
    static_assert(
        noexcept(std::declval<const AdamW&>().second_moment(std::declval<std::size_t>())),
        "second_moment is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(apply_step_schedule(std::declval<const RateSchedule&>(),
                std::declval<Optimizer&>())),
        "apply_step_schedule is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<const CosineSchedule&>().rate_at(std::declval<std::int64_t>())),
        "rate_at is noexcept");
    static_assert(
        noexcept(validate_cosine_schedule(std::declval<const CosineSchedule&>())),
        "validate_cosine_schedule is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(read_cosine_schedule_section(std::declval<const Config::Value&>())),
        "read_cosine_schedule_section is noexcept");
    static_assert(
        noexcept(write_cosine_schedule_section(std::declval<const CosineSchedule&>())),
        "write_cosine_schedule_section is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<GradientClip&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<DType>(), std::declval<std::size_t>(), std::declval<double>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<GradientClip&>().record(std::declval<SushiBLAS::TaskRecorder&>(),
                std::declval<const std::vector<SushiBLAS::Tensor*>&>())),
        "record is noexcept");
    static_assert(
        noexcept(std::declval<GradientClip&>().before_replay(std::declval<float>())),
        "before_replay is noexcept");
    static_assert(noexcept(std::declval<const GradientClip&>().scale()), "scale is noexcept");
    static_assert(noexcept(std::declval<const GradientClip&>().max_norm()), "max_norm is noexcept");
    static_assert(
        noexcept(std::declval<GradientClip&>().set_max_norm(std::declval<double>())),
        "set_max_norm is noexcept");
    static_assert(
        noexcept(std::declval<const GradientClip&>().applied_scale()),
        "applied_scale is noexcept");
    static_assert(
        noexcept(std::declval<const GradientClip&>().gradient_count()),
        "gradient_count is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<LossScaleState&>().make(std::declval<LossScaleOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<LossScaleState&>().observe(std::declval<bool>())),
        "observe is noexcept");
    static_assert(
        noexcept(std::declval<LossScaleState&>().restore(std::declval<double>(),
                std::declval<std::uint64_t>())),
        "restore is noexcept");
    static_assert(noexcept(std::declval<const LossScaleState&>().scale()), "scale is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaleState&>().inverse_scale()),
        "inverse_scale is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaleState&>().last_step_skipped()),
        "last_step_skipped is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaleState&>().consecutive_good_steps()),
        "consecutive_good_steps is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaleState&>().skipped_steps()),
        "skipped_steps is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaleState&>().backoffs()),
        "backoffs is noexcept");
    static_assert(noexcept(std::declval<const LossScaleState&>().growths()), "growths is noexcept");
    static_assert(noexcept(std::declval<const LossScaleState&>().options()), "options is noexcept");
    static_assert(
        noexcept(std::declval<LossScaler&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SushiBLAS::Tensor>(), std::declval<std::size_t>(),
                std::declval<LossScaleOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<LossScaler&>().record_screens(std::declval<SushiBLAS::TaskRecorder&>(
                ), std::declval<const std::vector<SushiBLAS::Tensor*>&>())),
        "record_screens is noexcept");
    static_assert(
        noexcept(std::declval<LossScaler&>().before_replay()),
        "before_replay is noexcept");
    static_assert(noexcept(std::declval<LossScaler&>().after_replay()), "after_replay is noexcept");
    static_assert(
        noexcept(std::declval<LossScaler&>().restore(std::declval<double>(),
                std::declval<std::uint64_t>())),
        "restore is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().gate()), "gate is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaler&>().parameter_count()),
        "parameter_count is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().scale()), "scale is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaler&>().inverse_scale()),
        "inverse_scale is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaler&>().last_step_skipped()),
        "last_step_skipped is noexcept");
    static_assert(
        noexcept(std::declval<const LossScaler&>().skipped_steps()),
        "skipped_steps is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().backoffs()), "backoffs is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().growths()), "growths is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().state()), "state is noexcept");
    static_assert(noexcept(std::declval<const LossScaler&>().to_string()), "to_string is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(validate_slots(std::declval<SushiRuntime::span<const ParameterSlot>>())),
        "validate_slots is noexcept");
    static_assert(noexcept(std::declval<Optimizer&>().step()), "step is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().parameter_count()),
        "parameter_count is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().step_count()),
        "step_count is noexcept");
    static_assert(noexcept(std::declval<Optimizer&>().run_state()), "run_state is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().check_state_owner(std::declval<std::string_view>(
                ))),
        "check_state_owner is noexcept");
    static_assert(
        noexcept(std::declval<Optimizer&>().restore_scalars(
                std::declval<SushiRuntime::span<const StateScalar>>())),
        "restore_scalars is noexcept");
    static_assert(
        noexcept(std::declval<Optimizer&>().set_learning_rate(std::declval<double>())),
        "set_learning_rate is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().parameters()),
        "parameters is noexcept");
    static_assert(
        noexcept(std::declval<Optimizer&>().set_loss_scaler(std::declval<LossScaler*>())),
        "set_loss_scaler is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().loss_scaler()),
        "loss_scaler is noexcept");
    static_assert(
        noexcept(std::declval<Optimizer&>().enable_gradient_clip(std::declval<double>())),
        "enable_gradient_clip is noexcept");
    static_assert(
        noexcept(std::declval<Optimizer&>().set_clip_norm(std::declval<double>())),
        "set_clip_norm is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().gradient_clip()),
        "gradient_clip is noexcept");
    static_assert(
        noexcept(std::declval<const Optimizer&>().is_in_compiled_step()),
        "is_in_compiled_step is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(make_adamw(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::vector<ParameterSlot>>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_adamw is noexcept");
    static_assert(
        noexcept(make_sgd(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::vector<ParameterSlot>>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_sgd is noexcept");
    static_assert(
        noexcept(make_optimizer(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::string_view>(), std::declval<std::vector<ParameterSlot>>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_optimizer is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(rate_at(std::declval<const RateSchedule&>(), std::declval<std::int64_t>())),
        "rate_at is noexcept");
    static_assert(
        noexcept(base_rate_of(std::declval<const RateSchedule&>())),
        "base_rate_of is noexcept");
    static_assert(
        noexcept(validate_rate_schedule(std::declval<const RateSchedule&>())),
        "validate_rate_schedule is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(read_rate_schedule_section(std::declval<const Config::Value&>())),
        "read_rate_schedule_section is noexcept");
    static_assert(
        noexcept(write_rate_schedule_section(std::declval<const RateSchedule&>())),
        "write_rate_schedule_section is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(read_rate_schedule(std::declval<const Config::Value&>(),
                std::declval<const Config::Value&>())),
        "read_rate_schedule is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<SGD&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::vector<ParameterSlot>>(), std::declval<SGDOptions>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const SGD&>().options()), "options is noexcept");
    static_assert(noexcept(std::declval<SGD&>().options()), "options is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<StepScalars&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::size_t>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const StepScalars&>().device_pointer()),
        "device_pointer is noexcept");
    static_assert(
        noexcept(std::declval<StepScalars&>().write(std::declval<SushiRuntime::span<const float>>(
                ))),
        "write is noexcept");
    static_assert(noexcept(std::declval<const StepScalars&>().count()), "count is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(std::declval<const StepSchedule&>().rate_at(std::declval<std::int64_t>())),
        "rate_at is noexcept");
    static_assert(
        noexcept(validate_step_schedule(std::declval<const StepSchedule&>())),
        "validate_step_schedule is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Optim
{
    static_assert(
        noexcept(read_step_schedule_section(std::declval<const Config::Value&>())),
        "read_step_schedule_section is noexcept");
    static_assert(
        noexcept(write_step_schedule_section(std::declval<const StepSchedule&>())),
        "write_step_schedule_section is noexcept");
} // namespace SushiAI::Optim

namespace SushiAI::Profile
{
    static_assert(noexcept(mark_period(std::declval<Timeline>())), "mark_period is noexcept");
    static_assert(noexcept(timeline_name(std::declval<Timeline>())), "timeline_name is noexcept");
} // namespace SushiAI::Profile

namespace SushiAI::Profile
{
    static_assert(
        noexcept(is_zone_group(std::declval<std::string_view>())),
        "is_zone_group is noexcept");
} // namespace SushiAI::Profile

namespace SushiAI
{
    static_assert(
        noexcept(Shape(std::declval<std::initializer_list<int64_t>>())),
        "Shape is noexcept");
    static_assert(
        noexcept(Shape(std::declval<SushiRuntime::span<const int64_t>>())),
        "Shape is noexcept");
    static_assert(
        noexcept(std::declval<Shape&>().make(std::declval<SushiRuntime::span<const int64_t>>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const Shape&>().rank()), "rank is noexcept");
    static_assert(
        noexcept(std::declval<const Shape&>().dim(std::declval<int32_t>())),
        "dim is noexcept");
    static_assert(
        noexcept(std::declval<const Shape&>().num_elements()),
        "num_elements is noexcept");
    static_assert(noexcept(std::declval<const Shape&>().data()), "data is noexcept");
    static_assert(noexcept(std::declval<const Shape&>().to_string()), "to_string is noexcept");
    static_assert(noexcept(std::declval<const Shape&>().hash()), "hash is noexcept");
} // namespace SushiAI

namespace SushiAI
{
    static_assert(
        noexcept(shape_of(std::declval<const SushiBLAS::Tensor&>())),
        "shape_of is noexcept");
    static_assert(
        noexcept(checked_shape_of(std::declval<const SushiBLAS::Tensor&>())),
        "checked_shape_of is noexcept");
} // namespace SushiAI

namespace SushiAI
{
    static_assert(noexcept(version_string()), "version_string is noexcept");
} // namespace SushiAI

namespace SushiAI::IO
{
    static_assert(
        noexcept(save_checkpoint(std::declval<const std::filesystem::path&>(),
                std::declval<NN::ModelState&>())),
        "save_checkpoint is noexcept");
    static_assert(
        noexcept(load_checkpoint(std::declval<const std::filesystem::path&>(),
                std::declval<NN::ModelState&>())),
        "load_checkpoint is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(std::declval<const TensorRecord&>().payload_bytes()),
        "payload_bytes is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointWriter&>().create(std::declval<std::ostream&>(),
                std::declval<std::uint32_t>(), std::declval<std::uint32_t>())),
        "create is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointWriter&>().begin_run_state(std::declval<std::string_view>(
                ))),
        "begin_run_state is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointWriter&>().write(std::declval<const TensorRecord&>(),
                std::declval<const void*>())),
        "write is noexcept");
    static_assert(noexcept(std::declval<const CheckpointWriter&>().finish()), "finish is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointWriter&>().declared_count()),
        "declared_count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointWriter&>().written_count()),
        "written_count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointWriter&>().declared_run_state_count()),
        "declared_run_state_count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointWriter&>().written_run_state_count()),
        "written_run_state_count is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointReader&>().open(std::declval<std::istream&>())),
        "open is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointReader&>().version()),
        "version is noexcept");
    static_assert(noexcept(std::declval<const CheckpointReader&>().count()), "count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointReader&>().read_count()),
        "read_count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointReader&>().run_state_count()),
        "run_state_count is noexcept");
    static_assert(
        noexcept(std::declval<const CheckpointReader&>().run_state_read_count()),
        "run_state_read_count is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointReader&>().begin_run_state()),
        "begin_run_state is noexcept");
    static_assert(noexcept(std::declval<CheckpointReader&>().next()), "next is noexcept");
    static_assert(
        noexcept(std::declval<CheckpointReader&>().read_payload(std::declval<const TensorRecord&>(),
                std::declval<void*>())),
        "read_payload is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(renamed(std::declval<std::string_view>(),
                std::declval<SushiRuntime::span<const RenameRule>>())),
        "renamed is noexcept");
    static_assert(
        noexcept(import_weights(std::declval<const std::filesystem::path&>(),
                std::declval<NN::ModelState&>(),
                std::declval<SushiRuntime::span<const RenameRule>>())),
        "import_weights is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(compose_training_file(std::declval<const ImportedModel&>())),
        "compose_training_file is noexcept");
    static_assert(
        noexcept(write_imported_model(std::declval<const ImportedModel&>(),
                std::declval<const std::filesystem::path&>(), std::declval<ExistingFiles>())),
        "write_imported_model is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(std::declval<ImporterRegistry&>().add(std::declval<const ImporterEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const ImporterRegistry&>().find(std::declval<std::string_view>())),
        "find is noexcept");
    static_assert(noexcept(std::declval<const ImporterRegistry&>().rows()), "rows is noexcept");
    static_assert(
        noexcept(import_model(std::declval<const ImporterRegistry&>(),
                std::declval<std::string_view>(),
                std::declval<SushiRuntime::span<const std::filesystem::path>>())),
        "import_model is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(save_run_checkpoint(std::declval<const std::filesystem::path&>(),
                std::declval<NN::ModelState&>(), std::declval<Optim::Optimizer&>(),
                std::declval<const RunProgress&>())),
        "save_run_checkpoint is noexcept");
    static_assert(
        noexcept(load_run_checkpoint(std::declval<const std::filesystem::path&>(),
                std::declval<NN::ModelState&>(), std::declval<Optim::Optimizer&>(),
                std::declval<RunProgress*>())),
        "load_run_checkpoint is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::IO
{
    static_assert(
        noexcept(check_run_seed(std::declval<const RunProgress&>(), std::declval<std::uint64_t>(),
                std::declval<std::string_view>())),
        "check_run_seed is noexcept");
} // namespace SushiAI::IO

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<Add&>().create(std::declval<ShapeSpan>())),
        "create is noexcept");
    static_assert(
        noexcept(make_add_node(std::declval<ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_add_node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<AvgPool2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const AvgPool2d&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const AvgPool2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(BatchNorm2d(std::declval<int64_t>(), std::declval<NormOptions>())),
        "BatchNorm2d is noexcept");
    static_assert(noexcept(std::declval<BatchNorm2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<BatchNorm2d&>().trace_training(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_training is noexcept");
    static_assert(
        noexcept(std::declval<BatchNorm2d&>().trace_evaluation(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_evaluation is noexcept");
    static_assert(
        noexcept(std::declval<const BatchNorm2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_batch_norm2d_node(std::declval<ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_batch_norm2d_node is noexcept");
    static_assert(
        noexcept(updated_statistic(std::declval<double>(), std::declval<double>(),
                std::declval<double>(), std::declval<double>())),
        "updated_statistic is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(Buffer(std::declval<const Buffer&>())), "Buffer is noexcept");
    static_assert(noexcept(Buffer(std::declval<Buffer&&>())), "Buffer is noexcept");
    static_assert(
        noexcept(std::declval<Buffer&>().define(std::declval<Shape>(), std::declval<double>(),
                std::declval<DType>())),
        "define is noexcept");
    static_assert(
        noexcept(std::declval<Buffer&>().read(std::declval<const Graph::GraphBuilder&>(),
                std::declval<std::string_view>())),
        "read is noexcept");
    static_assert(
        noexcept(std::declval<const Buffer&>().id_in(std::declval<const Graph::Graph&>())),
        "id_in is noexcept");
    static_assert(noexcept(std::declval<const Buffer&>().state()), "state is noexcept");
    static_assert(noexcept(std::declval<const Buffer&>().declared()), "declared is noexcept");
    static_assert(noexcept(std::declval<const Buffer&>().shape()), "shape is noexcept");
    static_assert(noexcept(std::declval<const Buffer&>().dtype()), "dtype is noexcept");
    static_assert(
        noexcept(std::declval<const Buffer&>().initial_value()),
        "initial_value is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(BuiltRows(std::declval<const ModelGraph&>())), "BuiltRows is noexcept");
    static_assert(noexcept(std::declval<const BuiltRows&>().size()), "size is noexcept");
    static_assert(
        noexcept(std::declval<const BuiltRows&>().node(std::declval<std::size_t>())),
        "node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<ChannelSlice&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const ChannelSlice&>().forward(
                std::declval<const Graph::GraphBuilder&>(), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const ChannelSlice&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<Concat&>().create(std::declval<ShapeSpan>())),
        "create is noexcept");
    static_assert(
        noexcept(make_concat_node(std::declval<ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_concat_node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Conv2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Conv2d&>().same_padded(std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>(), std::declval<int64_t>())),
        "same_padded is noexcept");
    static_assert(
        noexcept(std::declval<Conv2d&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Conv2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<DepthwiseConv2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<DepthwiseConv2d&>().same_padded(std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>(), std::declval<int64_t>())),
        "same_padded is noexcept");
    static_assert(
        noexcept(std::declval<DepthwiseConv2d&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const DepthwiseConv2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Dropout&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Dropout&>().trace_training(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_training is noexcept");
    static_assert(
        noexcept(std::declval<Dropout&>().trace_evaluation(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_evaluation is noexcept");
    static_assert(
        noexcept(std::declval<const Dropout&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_dropout_node(std::declval<ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_dropout_node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Embedding&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Embedding&>().forward(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<Embedding&>().forward_incremental(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const Embedding&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(noexcept(find_embedding(std::declval<Node&>())), "find_embedding is noexcept");
    static_assert(
        noexcept(make_embedding_node(std::declval<ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_embedding_node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Flatten&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Flatten&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Flatten&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<GlobalAvgPool&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const GlobalAvgPool&>().forward(
                std::declval<const Graph::GraphBuilder&>(), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const GlobalAvgPool&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<const HostTableList&>().tables()), "tables is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Identity&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Identity&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Identity&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(calculate_gain(std::declval<Nonlinearity>(), std::declval<double>())),
        "calculate_gain is noexcept");
    static_assert(
        noexcept(constant(std::declval<SushiBLAS::RandomOps&>(), std::declval<SushiBLAS::Tensor&>(),
                std::declval<double>())),
        "constant is noexcept");
    static_assert(
        noexcept(normal(std::declval<SushiBLAS::RandomOps&>(), std::declval<SushiBLAS::Tensor&>(),
                std::declval<double>())),
        "normal is noexcept");
    static_assert(
        noexcept(kaiming_normal(std::declval<SushiBLAS::RandomOps&>(),
                std::declval<SushiBLAS::Tensor&>(), std::declval<int64_t>(),
                std::declval<double>())),
        "kaiming_normal is noexcept");
    static_assert(
        noexcept(kaiming_uniform(std::declval<SushiBLAS::RandomOps&>(),
                std::declval<SushiBLAS::Tensor&>(), std::declval<int64_t>(),
                std::declval<double>())),
        "kaiming_uniform is noexcept");
    static_assert(
        noexcept(xavier_normal(std::declval<SushiBLAS::RandomOps&>(),
                std::declval<SushiBLAS::Tensor&>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<double>())),
        "xavier_normal is noexcept");
    static_assert(
        noexcept(xavier_uniform(std::declval<SushiBLAS::RandomOps&>(),
                std::declval<SushiBLAS::Tensor&>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<double>())),
        "xavier_uniform is noexcept");
    static_assert(
        noexcept(zeros(std::declval<SushiBLAS::RandomOps&>(), std::declval<SushiBLAS::Tensor&>())),
        "zeros is noexcept");
    static_assert(
        noexcept(initialize(std::declval<SushiBLAS::Engine&>(), std::declval<ModelState&>(),
                std::declval<const InitPolicy&>())),
        "initialize is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<const Layer&>().output_features()),
        "output_features is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(LayerNode(std::declval<std::unique_ptr<Layer>>())),
        "LayerNode is noexcept");
    static_assert(noexcept(std::declval<LayerNode&>().layer()), "layer is noexcept");
    static_assert(noexcept(std::declval<const LayerNode&>().layer()), "layer is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<LayerNorm&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<LayerNorm&>().forward(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<LayerNorm&>().forward_incremental(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const LayerNorm&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_layer_norm_node(std::declval<ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_layer_norm_node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(make_linear(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_linear is noexcept");
    static_assert(
        noexcept(make_gelu(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_gelu is noexcept");
    static_assert(
        noexcept(make_relu(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_relu is noexcept");
    static_assert(
        noexcept(make_tanh(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_tanh is noexcept");
    static_assert(
        noexcept(make_sigmoid(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_sigmoid is noexcept");
    static_assert(
        noexcept(make_leaky_relu(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_leaky_relu is noexcept");
    static_assert(
        noexcept(make_unflatten(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_unflatten is noexcept");
    static_assert(
        noexcept(make_conv2d(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_conv2d is noexcept");
    static_assert(
        noexcept(make_maxpool2d(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_maxpool2d is noexcept");
    static_assert(
        noexcept(make_avgpool2d(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_avgpool2d is noexcept");
    static_assert(
        noexcept(make_global_avgpool(std::declval<const Shape&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_global_avgpool is noexcept");
    static_assert(
        noexcept(make_flatten(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_flatten is noexcept");
    static_assert(
        noexcept(make_upsample(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_upsample is noexcept");
    static_assert(
        noexcept(make_silu(std::declval<const Shape&>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_silu is noexcept");
    static_assert(
        noexcept(register_layers(std::declval<NodeRegistry&>())),
        "register_layers is noexcept");
    static_assert(
        noexcept(build_layer_stack(std::declval<const NodeRegistry&>(),
                std::declval<const Config::ModelSpec&>())),
        "build_layer_stack is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<LayerStack&>().push(std::declval<std::unique_ptr<Layer>>())),
        "push is noexcept");
    static_assert(noexcept(std::declval<LayerStack&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<LayerStack&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<LayerStack&>().walk_parameters(std::declval<std::string_view>(),
                std::declval<const ParameterVisitor&>())),
        "walk_parameters is noexcept");
    static_assert(noexcept(std::declval<const LayerStack&>().size()), "size is noexcept");
    static_assert(noexcept(std::declval<const LayerStack&>().empty()), "empty is noexcept");
    static_assert(
        noexcept(std::declval<const LayerStack&>().output_features()),
        "output_features is noexcept");
    static_assert(
        noexcept(std::declval<const LayerStack&>().output_shape()),
        "output_shape is noexcept");
    static_assert(
        noexcept(std::declval<const LayerStack&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(std::declval<const LayerStack&>().layer(std::declval<std::size_t>())),
        "layer is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Linear&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Linear&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Linear&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(noexcept(std::declval<ReLU&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const ReLU&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const ReLU&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(noexcept(std::declval<GELU&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const GELU&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const GELU&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(noexcept(std::declval<Tanh&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Tanh&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Tanh&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(noexcept(std::declval<Sigmoid&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Sigmoid&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Sigmoid&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<LeakyReLU&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const LeakyReLU&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const LeakyReLU&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<const CrossEntropyLoss&>().forward(
                std::declval<const Graph::GraphBuilder&>(), std::declval<const Graph::Traced&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<MaxPool2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const MaxPool2d&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const MaxPool2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Mish&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Mish&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Mish&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<const NodeRef&>().node()), "node is noexcept");
    static_assert(noexcept(std::declval<const NodeRef&>().port()), "port is noexcept");
    static_assert(noexcept(std::declval<const NodeRef&>().is_input()), "is_input is noexcept");
    static_assert(noexcept(ModelGraph(std::declval<Shape>())), "ModelGraph is noexcept");
    static_assert(noexcept(std::declval<const ModelGraph&>().input()), "input is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().add(std::declval<std::string>(),
                std::declval<std::unique_ptr<Node>>(),
                std::declval<SushiRuntime::span<const NodeRef>>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const ModelGraph&>().port(std::declval<NodeRef>(),
                std::declval<std::string_view>())),
        "port is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().output(std::declval<std::string>(),
                std::declval<NodeRef>())),
        "output is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().output_ports(std::declval<NodeRef>())),
        "output_ports is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().trace(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().forward(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().trace_incremental(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>())),
        "trace_incremental is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().forward_incremental(std::declval<const Trace&>(),
                std::declval<Graph::ValueID>(), std::declval<const Shape&>())),
        "forward_incremental is noexcept");
    static_assert(
        noexcept(std::declval<const ModelGraph&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(std::declval<ModelGraph&>().walk_state(std::declval<std::string_view>(),
                std::declval<const StateVisitor&>())),
        "walk_state is noexcept");
    static_assert(noexcept(std::declval<ModelGraph&>().fields()), "fields is noexcept");
    static_assert(noexcept(std::declval<const ModelGraph&>().size()), "size is noexcept");
    static_assert(
        noexcept(std::declval<const ModelGraph&>().shape_of(std::declval<NodeRef>())),
        "shape_of is noexcept");
    static_assert(
        noexcept(std::declval<const ModelGraph&>().node(std::declval<std::size_t>())),
        "node is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<ModelOutputs&>().add(std::declval<std::string>(),
                std::declval<Graph::ValueID>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const ModelOutputs&>().at(std::declval<std::string_view>())),
        "at is noexcept");
    static_assert(noexcept(std::declval<const ModelOutputs&>().only()), "only is noexcept");
    static_assert(noexcept(std::declval<const ModelOutputs&>().all()), "all is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<ModelState&>().from_entries(std::declval<SushiBLAS::Engine&>(),
                std::declval<std::vector<StateEntry>>())),
        "from_entries is noexcept");
    static_assert(noexcept(ModelState(std::declval<ModelState&&>())), "ModelState is noexcept");
    static_assert(
        noexcept(std::declval<const ModelState&>().attach(std::declval<const Graph::Graph&>(),
                std::declval<Graph::TensorPool&>())),
        "attach is noexcept");
    static_assert(noexcept(std::declval<const ModelState&>().entries()), "entries is noexcept");
    static_assert(
        noexcept(std::declval<ModelState&>().tensor(std::declval<std::string_view>())),
        "tensor is noexcept");
    static_assert(
        noexcept(std::declval<ModelState&>().tensor(std::declval<Graph::StateID>())),
        "tensor is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<ParameterInit&>().fan_scaled()), "fan_scaled is noexcept");
    static_assert(
        noexcept(std::declval<ParameterInit&>().constant_value(std::declval<double>())),
        "constant_value is noexcept");
    static_assert(
        noexcept(std::declval<ParameterInit&>().normal(std::declval<double>())),
        "normal is noexcept");
    static_assert(
        noexcept(std::declval<Parameter&>().declare(std::declval<const Graph::GraphBuilder&>(),
                std::declval<Shape>(), std::declval<ParameterRole>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<std::string_view>(), std::declval<DType>())),
        "declare is noexcept");
    static_assert(
        noexcept(std::declval<Parameter&>().declare(std::declval<const Graph::GraphBuilder&>(),
                std::declval<Shape>(), std::declval<ParameterRole>(), std::declval<ParameterInit>(),
                std::declval<int64_t>(), std::declval<int64_t>(), std::declval<std::string_view>(),
                std::declval<DType>())),
        "declare is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().declared()), "declared is noexcept");
    static_assert(
        noexcept(std::declval<const Parameter&>().id_in(std::declval<const Graph::Graph&>())),
        "id_in is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().state()), "state is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().shape()), "shape is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().dtype()), "dtype is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().role()), "role is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().init()), "init is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().fan_in()), "fan_in is noexcept");
    static_assert(noexcept(std::declval<const Parameter&>().fan_out()), "fan_out is noexcept");
    static_assert(
        noexcept(std::declval<const StateVisitor&>().parameter(std::declval<const std::string&>(),
                std::declval<Parameter&>())),
        "parameter is noexcept");
    static_assert(
        noexcept(std::declval<const StateVisitor&>().buffer(std::declval<const std::string&>(),
                std::declval<Buffer&>())),
        "buffer is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<NodeRegistry&>().add(std::declval<const NodeEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<NodeRegistry&>().add(std::declval<const LayerEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<NodeRegistry&>().add(std::declval<const LinkedNodeEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const NodeRegistry&>().find_node(std::declval<std::string_view>())),
        "find_node is noexcept");
    static_assert(
        noexcept(std::declval<const NodeRegistry&>().find_linked(std::declval<std::string_view>())),
        "find_linked is noexcept");
    static_assert(
        noexcept(std::declval<const NodeRegistry&>().find_layer(std::declval<std::string_view>())),
        "find_layer is noexcept");
    static_assert(noexcept(std::declval<const NodeRegistry&>().nodes()), "nodes is noexcept");
    static_assert(noexcept(std::declval<const NodeRegistry&>().linked()), "linked is noexcept");
    static_assert(noexcept(std::declval<const NodeRegistry&>().layers()), "layers is noexcept");
    static_assert(
        noexcept(make_node(std::declval<const NodeRegistry&>(), std::declval<std::string_view>(),
                std::declval<ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>(), std::declval<const BuiltRows&>())),
        "make_node is noexcept");
    static_assert(
        noexcept(build_model(std::declval<const NodeRegistry&>(),
                std::declval<const Config::ModelSpec&>())),
        "build_model is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<SiLU&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const SiLU&>().forward(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const SiLU&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(
        noexcept(std::declval<const BufferUpdateList&>().updates()),
        "updates is noexcept");
    static_assert(
        noexcept(std::declval<const BufferUpdateList&>().sources()),
        "sources is noexcept");
    static_assert(noexcept(discard_updates()), "discard_updates is noexcept");
    static_assert(noexcept(std::declval<const FedLeafList&>().leaves()), "leaves is noexcept");
    static_assert(noexcept(std::declval<const CarriedValueList&>().values()), "values is noexcept");
    static_assert(
        noexcept(std::declval<const CarriedValueList&>().sources()),
        "sources is noexcept");
    static_assert(
        noexcept(std::declval<Trace&>().training(std::declval<const Graph::GraphBuilder&>(),
                std::declval<BufferUpdateSink&>(), std::declval<std::uint64_t>())),
        "training is noexcept");
    static_assert(
        noexcept(std::declval<Trace&>().evaluation(std::declval<const Graph::GraphBuilder&>())),
        "evaluation is noexcept");
    static_assert(
        noexcept(std::declval<Trace&>().carrying(std::declval<const Graph::GraphBuilder&>(),
                std::declval<FedLeafSink&>(), std::declval<CarriedValueSink&>())),
        "carrying is noexcept");
    static_assert(noexcept(std::declval<const Trace&>().builder()), "builder is noexcept");
    static_assert(noexcept(std::declval<const Trace&>().mode()), "mode is noexcept");
    static_assert(noexcept(std::declval<const Trace&>().seed()), "seed is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().claim_random_site()),
        "claim_random_site is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().fail(std::declval<Status>())),
        "fail is noexcept");
    static_assert(noexcept(std::declval<const Trace&>().status()), "status is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().value_or_fail(
                std::declval<const Result<Graph::ValueID>&>(), std::declval<Graph::ValueID>())),
        "value_or_fail is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().value_or_fail(std::declval<const Graph::Traced&>(),
                std::declval<Graph::ValueID>())),
        "value_or_fail is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().update(std::declval<Buffer&>(),
                std::declval<Graph::ValueID>(), std::declval<double>(), std::declval<double>())),
        "update is noexcept");
    static_assert(noexcept(std::declval<const Trace&>().is_carrying()), "is_carrying is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().feed(std::declval<std::string_view>(),
                std::declval<Shape>(), std::declval<DType>())),
        "feed is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().carried_leaf(std::declval<Shape>(),
                std::declval<std::string_view>(), std::declval<DType>())),
        "carried_leaf is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().carry(std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>(), std::declval<double>())),
        "carry is noexcept");
    static_assert(
        noexcept(std::declval<Trace&>().collect_tables(std::declval<HostTableSink&>())),
        "collect_tables is noexcept");
    static_assert(
        noexcept(std::declval<const Trace&>().host_table(std::declval<Shape>(),
                std::declval<std::string_view>(), std::declval<DType>(),
                std::declval<std::vector<double>>())),
        "host_table is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Unflatten&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Unflatten&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Unflatten&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::NN
{
    static_assert(noexcept(std::declval<Upsample2d&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<const Upsample2d&>().forward(std::declval<const Graph::GraphBuilder&>(
                ), std::declval<const Graph::Traced&>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Upsample2d&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::NN

namespace SushiAI::Train
{
    static_assert(
        noexcept(require_batch_role(std::declval<const Data::Batch&>(),
                std::declval<std::string_view>())),
        "require_batch_role is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(count_confusion(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<const SushiBLAS::Tensor&>(), std::declval<const ColumnOfLogits&>(),
                std::declval<double>())),
        "count_confusion is noexcept");
    static_assert(
        noexcept(accuracy_of(std::declval<const BinaryConfusion&>())),
        "accuracy_of is noexcept");
    static_assert(
        noexcept(precision_of(std::declval<const BinaryConfusion&>())),
        "precision_of is noexcept");
    static_assert(
        noexcept(recall_of(std::declval<const BinaryConfusion&>())),
        "recall_of is noexcept");
    static_assert(noexcept(f1_of(std::declval<const BinaryConfusion&>())), "f1_of is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<BinaryCrossEntropyObjective&>().make(std::declval<std::int64_t>(),
                std::declval<double>(), std::declval<DType>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const BinaryCrossEntropyObjective&>().batch()),
        "batch is noexcept");
    static_assert(
        noexcept(std::declval<const BinaryCrossEntropyObjective&>().threshold()),
        "threshold is noexcept");
    static_assert(
        noexcept(make_binary_cross_entropy(std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_binary_cross_entropy is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(bind_buffer_updates(std::declval<const NN::BufferUpdateList&>(),
                std::declval<NN::ModelState&>(), std::declval<const Graph::TensorPool&>())),
        "bind_buffer_updates is noexcept");
    static_assert(
        noexcept(BufferUpdateRecorder(std::declval<std::vector<BufferUpdateSlot>>())),
        "BufferUpdateRecorder is noexcept");
    static_assert(
        noexcept(std::declval<const BufferUpdateRecorder&>().slot_count()),
        "slot_count is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(validate_threshold(std::declval<double>())),
        "validate_threshold is noexcept");
    static_assert(
        noexcept(read_threshold(std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "read_threshold is noexcept");
    static_assert(noexcept(logit_of(std::declval<double>())), "logit_of is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(emit_mean_of_elements(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<const Graph::Traced&>())),
        "emit_mean_of_elements is noexcept");
    static_assert(
        noexcept(emit_logistic_loss(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>())),
        "emit_logistic_loss is noexcept");
    static_assert(
        noexcept(emit_squared_error(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>())),
        "emit_squared_error is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(fill_dense_targets(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<SushiBLAS::Tensor&>())),
        "fill_dense_targets is noexcept");
    static_assert(
        noexcept(fill_binary_targets(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<SushiBLAS::Tensor&>())),
        "fill_binary_targets is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<EvaluationStep&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const TracedStep&>(), std::declval<const NN::ModelState&>(),
                std::declval<Graph::TensorPool&>(), std::declval<const Graph::FusionSelector*>(),
                std::declval<const Graph::TensorPool*>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<EvaluationStep&>().run(std::declval<const SushiBLAS::Tensor&>())),
        "run is noexcept");
    static_assert(
        noexcept(std::declval<EvaluationStep&>().run_in_order(std::declval<FeedSpan>())),
        "run_in_order is noexcept");
    static_assert(
        noexcept(std::declval<EvaluationStep&>().run_by_name(
                std::declval<SushiRuntime::span<const NamedFeed>>())),
        "run_by_name is noexcept");
    static_assert(noexcept(std::declval<EvaluationStep&>().reset()), "reset is noexcept");
    static_assert(
        noexcept(std::declval<const EvaluationStep&>().carries_state()),
        "carries_state is noexcept");
    static_assert(
        noexcept(std::declval<const EvaluationStep&>().input_names()),
        "input_names is noexcept");
    static_assert(
        noexcept(std::declval<const EvaluationStep&>().input(std::declval<std::string_view>())),
        "input is noexcept");
    static_assert(
        noexcept(std::declval<const EvaluationStep&>().output(std::declval<std::string_view>())),
        "output is noexcept");
    static_assert(noexcept(std::declval<const EvaluationStep&>().step()), "step is noexcept");
    static_assert(
        noexcept(std::declval<EvaluationStep&>().restore_run_count(std::declval<std::size_t>())),
        "restore_run_count is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(check_feed(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<const SushiBLAS::Tensor&>(), std::declval<std::string_view>())),
        "check_feed is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(require_host_reals(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::int64_t>(), std::declval<std::string_view>())),
        "require_host_reals is noexcept");
    static_assert(
        noexcept(host_real(std::declval<const SushiBLAS::Tensor&>(), std::declval<std::int64_t>())),
        "host_real is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(describe_input_leaf(std::declval<const Graph::Value&>())),
        "describe_input_leaf is noexcept");
    static_assert(
        noexcept(match_input(std::declval<const Data::InputDescription&>(),
                std::declval<const Data::Dataset&>())),
        "match_input is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<MeanSquaredErrorObjective&>().make(std::declval<std::int64_t>(),
                std::declval<std::int64_t>(), std::declval<DType>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const MeanSquaredErrorObjective&>().batch()),
        "batch is noexcept");
    static_assert(
        noexcept(std::declval<const MeanSquaredErrorObjective&>().width()),
        "width is noexcept");
    static_assert(
        noexcept(make_mean_squared_error(std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_mean_squared_error is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<Metrics&>().add(std::declval<const Metric&>())),
        "add is noexcept");
    static_assert(noexcept(std::declval<Metrics&>().clear()), "clear is noexcept");
    static_assert(
        noexcept(std::declval<const Metrics&>().find(std::declval<std::string_view>())),
        "find is noexcept");
    static_assert(
        noexcept(std::declval<const Metrics&>().value_or(std::declval<std::string_view>(),
                std::declval<double>())),
        "value_or is noexcept");
    static_assert(noexcept(std::declval<const Metrics&>().all()), "all is noexcept");
    static_assert(
        noexcept(describe_metric_value(std::declval<const Metric&>())),
        "describe_metric_value is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<MultiLabelObjective&>().make(std::declval<std::int64_t>(),
                std::declval<std::int64_t>(), std::declval<double>(), std::declval<DType>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const MultiLabelObjective&>().batch()),
        "batch is noexcept");
    static_assert(
        noexcept(std::declval<const MultiLabelObjective&>().classes()),
        "classes is noexcept");
    static_assert(
        noexcept(std::declval<const MultiLabelObjective&>().threshold()),
        "threshold is noexcept");
    static_assert(
        noexcept(make_multi_label(std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_multi_label is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<ObjectiveLeaves&>().add(std::declval<std::string_view>(),
                std::declval<Graph::ValueID>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const ObjectiveLeaves&>().has(std::declval<std::string_view>())),
        "has is noexcept");
    static_assert(
        noexcept(std::declval<const ObjectiveLeaves&>().at(std::declval<std::string_view>())),
        "at is noexcept");
    static_assert(noexcept(std::declval<const ObjectiveLeaves&>().all()), "all is noexcept");
    static_assert(
        noexcept(std::declval<ObjectiveLeaves&>().require(std::declval<Data::TargetDescription>())),
        "require is noexcept");
    static_assert(
        noexcept(std::declval<const ObjectiveLeaves&>().required()),
        "required is noexcept");
    static_assert(
        noexcept(find_leaf_tensor(std::declval<const ObjectiveLeaves&>(),
                std::declval<std::string_view>(), std::declval<Graph::TensorPool&>())),
        "find_leaf_tensor is noexcept");
    static_assert(
        noexcept(CrossEntropyObjective(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<DType>())),
        "CrossEntropyObjective is noexcept");
    static_assert(
        noexcept(make_cross_entropy(std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_cross_entropy is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(list_output_shapes(std::declval<const Graph::Graph&>(),
                std::declval<const NN::ModelOutputs&>())),
        "list_output_shapes is noexcept");
    static_assert(
        noexcept(find_output_shape(std::declval<const std::vector<OutputShape>&>(),
                std::declval<std::string_view>())),
        "find_output_shape is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<ObjectiveRegistry&>().add(std::declval<const ObjectiveEntry&>())),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<const ObjectiveRegistry&>().find(std::declval<std::string_view>())),
        "find is noexcept");
    static_assert(noexcept(std::declval<const ObjectiveRegistry&>().rows()), "rows is noexcept");
    static_assert(
        noexcept(register_objectives(std::declval<ObjectiveRegistry&>())),
        "register_objectives is noexcept");
    static_assert(
        noexcept(make_objective(std::declval<const ObjectiveRegistry&>(),
                std::declval<std::string_view>(), std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_objective is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(fill_one_hot(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<SushiBLAS::Tensor&>())),
        "fill_one_hot is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(takes_weight_decay(std::declval<NN::ParameterRole>())),
        "takes_weight_decay is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(match_roles(std::declval<const std::vector<Data::TargetDescription>&>(),
                std::declval<const Data::Dataset&>())),
        "match_roles is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(only_output(std::declval<const StepView&>())),
        "only_output is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(batch_shape(std::declval<int64_t>(), std::declval<const Shape&>())),
        "batch_shape is noexcept");
    static_assert(
        noexcept(retain(std::declval<const NN::BufferUpdateList&>(),
                std::declval<std::vector<Graph::ValueID>&>(),
                std::declval<std::vector<Graph::ValueID>&>())),
        "retain is noexcept");
    static_assert(
        noexcept(fill_host_tables(std::declval<const NN::HostTableList&>(),
                std::declval<Graph::TensorPool&>())),
        "fill_host_tables is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<SumObjective&>().make(std::declval<std::vector<SumTerm>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<SumObjective&>().leaf_prefix(std::declval<std::size_t>())),
        "leaf_prefix is noexcept");
    static_assert(noexcept(std::declval<const SumObjective&>().terms()), "terms is noexcept");
    static_assert(
        noexcept(make_sum_objective(std::declval<const ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_sum_objective is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(bind_parameters(std::declval<const TrainingGraph&>(),
                std::declval<const NN::ModelState&>(), std::declval<Graph::TensorPool&>(),
                std::declval<Graph::PlanOptions>())),
        "bind_parameters is noexcept");
    static_assert(
        noexcept(bind_step_results(std::declval<const TrainingGraph&>(),
                std::declval<const NN::ModelState&>(), std::declval<Graph::TensorPool&>())),
        "bind_step_results is noexcept");
    static_assert(noexcept(std::declval<const RunReport&>().to_string()), "to_string is noexcept");
    static_assert(
        noexcept(std::declval<Trainer&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const TrainingGraph&>(), std::declval<Graph::TensorPool&>(),
                std::declval<const Objective&>(), std::declval<Optim::Optimizer&>(),
                std::declval<TrainerOptions>(), std::declval<std::vector<BufferUpdateSlot>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<Trainer&>().run_step(std::declval<const Data::Batch&>())),
        "run_step is noexcept");
    static_assert(
        noexcept(std::declval<Trainer&>().fit(std::declval<const Data::Dataset&>())),
        "fit is noexcept");
    static_assert(
        noexcept(std::declval<const Trainer&>().compile_count()),
        "compile_count is noexcept");
    static_assert(noexcept(std::declval<const Trainer&>().step_count()), "step_count is noexcept");
    static_assert(
        noexcept(std::declval<const Trainer&>().optimizer_is_in_plan()),
        "optimizer_is_in_plan is noexcept");
    static_assert(noexcept(std::declval<const Trainer&>().step()), "step is noexcept");
    static_assert(
        noexcept(std::declval<Trainer&>().restore_run_count(std::declval<std::size_t>())),
        "restore_run_count is noexcept");
    static_assert(
        noexcept(std::declval<const Trainer&>().current_loss()),
        "current_loss is noexcept");
    static_assert(noexcept(std::declval<Trainer&>().options()), "options is noexcept");
} // namespace SushiAI::Train

namespace SushiAI::Train
{
    static_assert(
        noexcept(std::declval<TrainingStep&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::TensorPool&>(),
                std::declval<Optim::Optimizer&>(), std::declval<std::vector<BufferUpdateSlot>>(),
                std::declval<TrainingStepOptions>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<TrainingStep&>().run()), "run is noexcept");
    static_assert(
        noexcept(std::declval<const TrainingStep&>().has_host_stage()),
        "has_host_stage is noexcept");
    static_assert(
        noexcept(std::declval<TrainingStep&>().restore_run_count(std::declval<std::size_t>())),
        "restore_run_count is noexcept");
    static_assert(noexcept(std::declval<const TrainingStep&>().lead()), "lead is noexcept");
    static_assert(
        noexcept(std::declval<const TrainingStep&>().optimizer_is_in_plan()),
        "optimizer_is_in_plan is noexcept");
    static_assert(noexcept(std::declval<TrainingStep&>().optimizer()), "optimizer is noexcept");
    static_assert(noexcept(std::declval<TrainingStep&>().step()), "step is noexcept");
    static_assert(noexcept(std::declval<const TrainingStep&>().step()), "step is noexcept");
} // namespace SushiAI::Train

#if defined(SA_TEST_DISTRIBUTED_BUILD)
namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<ClusterSection&>().read(std::declval<const Config::Value&>(),
                std::declval<const std::filesystem::path&>())),
        "read is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().listen_address()),
        "listen_address is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().secret_file()),
        "secret_file is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().attach_timeout()),
        "attach_timeout is noexcept");
    static_assert(noexcept(std::declval<const ClusterSection&>().plan()), "plan is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().worker_of(std::declval<std::size_t>())),
        "worker_of is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().absent_policy_of(std::declval<std::size_t>(
                ))),
        "absent_policy_of is noexcept");
    static_assert(
        noexcept(std::declval<const ClusterSection&>().placement_rules()),
        "placement_rules is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(sums_exactly(std::declval<SushiBLAS::Core::DataType>())),
        "sums_exactly is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<GradientExchange&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SushiBLAS::ClusterBinding&>(), std::declval<const ReplicaPlan&>(),
                std::declval<const ExchangeRun&>(), std::declval<const SushiBLAS::TensorPack&>(),
                std::declval<const SushiBLAS::PackLayout&>(),
                std::declval<std::vector<SushiBLAS::Tensor>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const GradientExchange&>().replica_count()),
        "replica_count is noexcept");
    static_assert(
        noexcept(std::declval<const GradientExchange&>().next_step()),
        "next_step is noexcept");
    static_assert(
        noexcept(std::declval<GradientExchange&>().set_next_step(std::declval<std::uint64_t>())),
        "set_next_step is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(is_host_visible(std::declval<const SushiBLAS::Tensor&>())),
        "is_host_visible is noexcept");
    static_assert(
        noexcept(require_host_visible(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::string_view>(), std::declval<std::size_t>())),
        "require_host_visible is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<ParameterPublish&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SushiBLAS::ClusterBinding&>(),
                std::declval<const SushiBLAS::PackLayout&>(),
                std::declval<std::vector<SushiBLAS::Tensor>>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const ParameterPublish&>().pack()), "pack is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(replica_seed(std::declval<const ReplicaDraws&>())),
        "replica_seed is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<ReplicaKernel&>().make(std::declval<ReplicaPlan>(),
                std::declval<ReplicaModelFactory>(), std::declval<const ReplicaDraws&>(),
                std::declval<const SushiRuntime::RuntimeConfig&>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaKernel&>().operation_name()),
        "operation_name is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaKernel&>().operation()),
        "operation is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaKernel&>().parameter_layout()),
        "parameter_layout is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaKernel&>().result_layout()),
        "result_layout is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaKernel&>().observe(std::declval<ReplicaServeObserver*>())),
        "observe is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaKernel&>().serve(
                std::declval<SushiRuntime::span<const SushiRuntime::Distributed::BufferView>>(),
                std::declval<SushiRuntime::span<const SushiRuntime::Distributed::BufferView>>(),
                std::declval<const SushiRuntime::NodeOptions&>())),
        "serve is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<ReplicaPlan&>().make(std::declval<std::vector<ReplicaShare>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().replica_count()),
        "replica_count is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().batch_rows()),
        "batch_rows is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().rows_of(std::declval<std::size_t>())),
        "rows_of is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().micro_batch_of(std::declval<std::size_t>())),
        "micro_batch_of is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().micro_batch_count_of(std::declval<std::size_t>(
                ))),
        "micro_batch_count_of is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().weight_of(std::declval<std::size_t>())),
        "weight_of is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().rows(std::declval<std::size_t>(),
                std::declval<std::uint64_t>())),
        "rows is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaPlan&>().micro_batch(std::declval<std::size_t>(),
                std::declval<std::uint64_t>(), std::declval<std::uint64_t>())),
        "micro_batch is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(
        noexcept(std::declval<const ReplicaPosition&>().draw_index(std::declval<std::uint64_t>(),
                std::declval<std::uint64_t>())),
        "draw_index is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaPosition&>().at_step(std::declval<std::uint64_t>(),
                std::declval<std::uint64_t>())),
        "at_step is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaPosition&>().read(
                std::declval<const SushiRuntime::Distributed::BufferView&>())),
        "read is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(noexcept(replica_runtime_config()), "replica_runtime_config is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaStep&>().make(std::declval<const ReplicaModelFactory&>(),
                std::declval<const ReplicaDraws&>(), std::declval<std::int64_t>(),
                std::declval<const SushiRuntime::RuntimeConfig&>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaStep&>().parameter_layout()),
        "parameter_layout is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaStep&>().result_layout()),
        "result_layout is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaStep&>().fingerprint()),
        "fingerprint is noexcept");
    static_assert(noexcept(std::declval<const ReplicaStep&>().draws()), "draws is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaStep&>().begin(std::declval<const SushiBLAS::BoundPack&>())),
        "begin is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaStep&>().run_micro_batch(std::declval<const ReplicaRows&>())),
        "run_micro_batch is noexcept");
    static_assert(
        noexcept(std::declval<const ReplicaStep&>().finish(
                std::declval<const SushiBLAS::BoundPack&>())),
        "finish is noexcept");
} // namespace SushiAI::Distribution

namespace SushiAI::Distribution
{
    static_assert(noexcept(std::declval<const ReplicaTask&>().encode()), "encode is noexcept");
    static_assert(
        noexcept(std::declval<ReplicaTask&>().decode(
                std::declval<SushiRuntime::span<const std::uint64_t>>())),
        "decode is noexcept");
    static_assert(noexcept(std::declval<const ReplicaTask&>().draws()), "draws is noexcept");
} // namespace SushiAI::Distribution
#endif

/** @brief Passes when the file compiled: the assertions above are the test. */
TEST(Small_NoexceptSurface, EveryPublicFunctionIsNoexcept)
{
    SUCCEED();
}
