/**************************************************************************/
/* SushiAI.h                                                              */
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
 * @file SushiAI.h
 * @brief Includes the full public interface of the SushiAI library.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/version.hpp>

#include <SushiAI/config/value.hpp>
#include <SushiAI/config/registry.hpp>
#include <SushiAI/config/model_config.hpp>
#include <SushiAI/config/train_config.hpp>

#include <SushiAI/core/common.hpp>
#include <SushiAI/core/ops.hpp>
#include <SushiAI/core/shape.hpp>
#include <SushiAI/core/tensor_shape.hpp>
#include <SushiAI/core/allocation.hpp>
#include <SushiAI/core/check.hpp>
#include <SushiAI/core/status.hpp>
#include <SushiAI/core/result.hpp>
#include <SushiAI/core/seam.hpp>

#include <SushiAI/graph/ir.hpp>
#include <SushiAI/graph/builder.hpp>
#include <SushiAI/graph/precision.hpp>
#include <SushiAI/graph/cast_insertion.hpp>
#include <SushiAI/graph/fusion.hpp>
#include <SushiAI/graph/memory_plan.hpp>
#include <SushiAI/graph/arena.hpp>
#include <SushiAI/graph/tensor_pool.hpp>
#include <SushiAI/graph/fusion_selector.hpp>
#include <SushiAI/graph/lowering.hpp>
#include <SushiAI/graph/lowering/rule.hpp>
#include <SushiAI/graph/lowering/unlowered.hpp>
#include <SushiAI/graph/lowering/context.hpp>
#include <SushiAI/graph/lowering/elementwise.hpp>
#include <SushiAI/graph/lowering/linalg.hpp>
#include <SushiAI/graph/lowering/activation.hpp>
#include <SushiAI/graph/lowering/loss.hpp>
#include <SushiAI/graph/lowering/shape.hpp>
#include <SushiAI/graph/lowering/spatial.hpp>
#include <SushiAI/graph/lowering/norm.hpp>
#include <SushiAI/graph/lowering/fused.hpp>
#include <SushiAI/graph/lowering/all.hpp>
#include <SushiAI/graph/step_recorder.hpp>
#include <SushiAI/graph/compiled_step.hpp>
#include <SushiAI/graph/rematerialise.hpp>
#include <SushiAI/graph/traced.hpp>
#include <SushiAI/graph/state_id.hpp>

#include <SushiAI/autograd/backward.hpp>
#include <SushiAI/autograd/tape.hpp>
#include <SushiAI/autograd/provenance.hpp>
#include <SushiAI/autograd/rule.hpp>
#include <SushiAI/autograd/rules/all.hpp>
#include <SushiAI/autograd/rules/elementwise.hpp>
#include <SushiAI/autograd/rules/linalg.hpp>
#include <SushiAI/autograd/rules/activation.hpp>
#include <SushiAI/autograd/rules/loss.hpp>
#include <SushiAI/autograd/rules/shape.hpp>
#include <SushiAI/autograd/rules/spatial.hpp>
#include <SushiAI/autograd/rules/norm.hpp>
#include <SushiAI/autograd/gradcheck.hpp>

#include <SushiAI/nn/module.hpp>
#include <SushiAI/nn/layers.hpp>
#include <SushiAI/nn/unflatten.hpp>
#include <SushiAI/nn/conv2d.hpp>
#include <SushiAI/nn/max_pool2d.hpp>
#include <SushiAI/nn/avg_pool2d.hpp>
#include <SushiAI/nn/global_avg_pool.hpp>
#include <SushiAI/nn/flatten.hpp>
#include <SushiAI/nn/leaky_relu.hpp>
#include <SushiAI/nn/upsample2d.hpp>
#include <SushiAI/nn/layer.hpp>
#include <SushiAI/nn/layer_stack.hpp>
#include <SushiAI/nn/layer_registry.hpp>
#include <SushiAI/nn/loss.hpp>
#include <SushiAI/nn/init.hpp>
#include <SushiAI/nn/batch_norm2d.hpp>
#include <SushiAI/nn/buffer.hpp>
#include <SushiAI/nn/concat.hpp>
#include <SushiAI/nn/identity.hpp>
#include <SushiAI/nn/layer_node.hpp>
#include <SushiAI/nn/model_graph.hpp>
#include <SushiAI/nn/model_outputs.hpp>
#include <SushiAI/nn/model_state.hpp>
#include <SushiAI/nn/module_list.hpp>
#include <SushiAI/nn/module_node.hpp>
#include <SushiAI/nn/node.hpp>
#include <SushiAI/nn/node_registry.hpp>
#include <SushiAI/nn/node_attributes.hpp>
#include <SushiAI/nn/silu.hpp>
#include <SushiAI/nn/trace.hpp>
#include <SushiAI/nn/throwing_bridge.hpp>

#include <SushiAI/optim/step_scalars.hpp>
#include <SushiAI/optim/loss_scaler.hpp>
#include <SushiAI/optim/optimizer.hpp>
#include <SushiAI/optim/sgd.hpp>
#include <SushiAI/optim/adamw.hpp>
#include <SushiAI/optim/optimizer_registry.hpp>

#include <SushiAI/data/dataset.hpp>
#include <SushiAI/data/dataset_registry.hpp>
#include <SushiAI/data/materialised_dataset.hpp>

#include <SushiAI/eager/segment.hpp>
#include <SushiAI/eager/scope_trace.hpp>
#include <SushiAI/eager/scope.hpp>
#include <SushiAI/eager/partition.hpp>
#include <SushiAI/eager/region_partitioner.hpp>
#include <SushiAI/eager/scope_partitioner.hpp>
#include <SushiAI/eager/region_signature.hpp>

#include <SushiAI/io/checkpoint_format.hpp>
#include <SushiAI/io/import_weights.hpp>
#include <SushiAI/io/checkpoint.hpp>

#include <SushiAI/train/objective.hpp>
#include <SushiAI/train/training_graph.hpp>
#include <SushiAI/train/steps.hpp>
#include <SushiAI/train/buffer_update_recorder.hpp>
#include <SushiAI/train/trainer.hpp>
#include <SushiAI/train/training_step.hpp>
#include <SushiAI/train/feed.hpp>
#include <SushiAI/train/evaluation_step.hpp>
