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
 * @brief Tests that every public function of the vision modules is noexcept.
 * @author Mustafa Garip
 */

#include <utility>

#include <gtest/gtest.h>

#include <SushiAI/vision/blocks/attention.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c2f.hpp>
#include <SushiAI/vision/blocks/c2psa.hpp>
#include <SushiAI/vision/blocks/c3.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/blocks/psa_block.hpp>
#include <SushiAI/vision/blocks/segment_head.hpp>
#include <SushiAI/vision/blocks/sppf.hpp>
#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/codecs/image_format.hpp>
#include <SushiAI/vision/codecs/jpeg_decoder.hpp>
#include <SushiAI/vision/codecs/png_decoder.hpp>
#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/cifar.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/coco_json.hpp>
#include <SushiAI/vision/datasets/csv.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/idx.hpp>
#include <SushiAI/vision/datasets/idx_file.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>
#include <SushiAI/vision/datasets/image_folder.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/datasets/mask_png.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/polygon_mask.hpp>
#include <SushiAI/vision/datasets/reader_dataset.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample_cache.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/datasets/table_dataset.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>
#include <SushiAI/vision/datasets/voc_xml.hpp>
#include <SushiAI/vision/datasets/xml_element.hpp>
#include <SushiAI/vision/datasets/yolo_txt.hpp>
#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_count.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_plan.hpp>
#include <SushiAI/vision/detect/anchor/anchor_review.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/anchor/head_truth.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/batch_average_precision.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/float_order.hpp>
#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/assigner.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/free/target_leaves.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/detect/iou.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>
#include <SushiAI/vision/detect/top_k.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/flip.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/labels.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/imaging/random.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>
#include <SushiAI/vision/imaging/rectangle.hpp>
#include <SushiAI/vision/imaging/resize.hpp>
#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_fold.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_recipe.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_tensors.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>
#include <SushiAI/vision/registration/registration.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/anchor_training_loop.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <SushiAI/vision/training/detection_run_check.hpp>
#include <SushiAI/vision/training/detection_truth_check.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/dice_targets.hpp>
#include <SushiAI/vision/training/distribution_expectation_graph.hpp>
#include <SushiAI/vision/training/distribution_focal_loss.hpp>
#include <SushiAI/vision/training/free_assignment_stage.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_bias_init.hpp>
#include <SushiAI/vision/training/free_branch_weights.hpp>
#include <SushiAI/vision/training/free_detection_loss.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>
#include <SushiAI/vision/training/free_training_loop.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>
#include <SushiAI/vision/training/head_delta_rule_choice.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>
#include <SushiAI/vision/training/pixel_targets.hpp>
#include <SushiAI/vision/training/run_progress.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(Attention(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<NN::NormOptions>())),
        "Attention is noexcept");
    static_assert(noexcept(std::declval<Attention&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Attention&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Attention&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(Bottleneck(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<bool>(), std::declval<NN::NormOptions>(),
                std::declval<int64_t>(), std::declval<int64_t>())),
        "Bottleneck is noexcept");
    static_assert(noexcept(std::declval<Bottleneck&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<Bottleneck&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const Bottleneck&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_bottleneck_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_bottleneck_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(make_c2f_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_c2f_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(noexcept(std::declval<C2PSA&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<C2PSA&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const C2PSA&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_c2psa_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_c2psa_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(make_c3_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_c3_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(C3k(std::declval<int64_t>(), std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<bool>(), std::declval<NN::NormOptions>())),
        "C3k is noexcept");
    static_assert(
        noexcept(C3k(std::declval<int64_t>(), std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<bool>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<NN::NormOptions>())),
        "C3k is noexcept");
    static_assert(noexcept(std::declval<C3k&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<C3k&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const C3k&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_c3k_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_c3k_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(make_c3k2_of_bottleneck(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_c3k2_of_bottleneck is noexcept");
    static_assert(
        noexcept(make_c3k2_of_c3k(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_c3k2_of_c3k is noexcept");
    static_assert(
        noexcept(make_c3k2_of_bottleneck_psa(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_c3k2_of_bottleneck_psa is noexcept");
    static_assert(
        noexcept(make_c3k2_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_c3k2_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(make_conv_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_conv_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(std::declval<DetectHead&>().create(std::declval<NN::ShapeSpan>(),
                std::declval<int64_t>(), std::declval<NN::NormOptions>(),
                std::declval<const DetectOptions&>())),
        "create is noexcept");
    static_assert(
        noexcept(make_detect_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_detect_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(PSABlock(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<NN::NormOptions>())),
        "PSABlock is noexcept");
    static_assert(noexcept(std::declval<PSABlock&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<PSABlock&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const PSABlock&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(
        noexcept(SegmentHead(std::declval<int64_t>(), std::declval<int64_t>(),
                std::declval<int64_t>(), std::declval<int64_t>(), std::declval<NN::NormOptions>())),
        "SegmentHead is noexcept");
    static_assert(noexcept(std::declval<SegmentHead&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<SegmentHead&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const SegmentHead&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_segment_head_node(std::declval<NN::ShapeSpan>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_segment_head_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Blocks
{
    static_assert(noexcept(std::declval<SPPF&>().fields()), "fields is noexcept");
    static_assert(
        noexcept(std::declval<SPPF&>().forward(std::declval<const NN::Trace&>(),
                std::declval<Graph::ValueID>())),
        "forward is noexcept");
    static_assert(
        noexcept(std::declval<const SPPF&>().output_shape(std::declval<const Shape&>())),
        "output_shape is noexcept");
    static_assert(
        noexcept(make_sppf_node(std::declval<NN::ShapeSpan>(), std::declval<const Config::Value&>(),
                std::declval<std::string_view>())),
        "make_sppf_node is noexcept");
} // namespace SushiAI::Vision::Blocks

namespace SushiAI::Vision::Codecs
{
    static_assert(
        noexcept(decode_image(std::declval<SushiRuntime::span<const std::uint8_t>>(),
                std::declval<const DecodeOptions&>())),
        "decode_image is noexcept");
    static_assert(
        noexcept(load_image(std::declval<const std::filesystem::path&>(),
                std::declval<const DecodeOptions&>())),
        "load_image is noexcept");
} // namespace SushiAI::Vision::Codecs

namespace SushiAI::Vision::Codecs
{
    static_assert(
        noexcept(detect_image_format(std::declval<SushiRuntime::span<const std::uint8_t>>())),
        "detect_image_format is noexcept");
} // namespace SushiAI::Vision::Codecs

namespace SushiAI::Vision::Codecs
{
    static_assert(
        noexcept(decode_jpeg(std::declval<SushiRuntime::span<const std::uint8_t>>(),
                std::declval<const DecodeOptions&>())),
        "decode_jpeg is noexcept");
} // namespace SushiAI::Vision::Codecs

namespace SushiAI::Vision::Codecs
{
    static_assert(
        noexcept(decode_png(std::declval<SushiRuntime::span<const std::uint8_t>>(),
                std::declval<const DecodeOptions&>())),
        "decode_png is noexcept");
} // namespace SushiAI::Vision::Codecs

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<AugmentPipeline&>().make(std::declval<const AugmentOptions&>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(write_batch_inputs(std::declval<SushiRuntime::span<const FittedSample>>(),
                std::declval<SushiBLAS::Tensor&>())),
        "write_batch_inputs is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<BinaryCollator&>().make(std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(read_cifar(std::declval<const CIFAROptions&>())),
        "read_cifar is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<ClassificationCollator&>().make(std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(read_coco_json(std::declval<const COCOJSONOptions&>())),
        "read_coco_json is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(read_csv(std::declval<const CSVOptions&>())), "read_csv is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<DetectionCollator&>().make(std::declval<std::int32_t>(),
                std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(fit_mask(std::declval<Imaging::ImageView>(), std::declval<Imaging::ImageSize>(),
                std::declval<const SampleFitOptions&>(), std::declval<Imaging::Image&>())),
        "fit_mask is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<FitPipeline&>().make(std::declval<const SampleFitOptions&>(),
                std::declval<std::int32_t>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<FitPipeline&>().layout_of(std::declval<std::int32_t>())),
        "layout_of is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(fit_sample(std::declval<const Sample&>(), std::declval<Imaging::ImageView>(),
                std::declval<const SampleFitOptions&>(), std::declval<FittedSample&>())),
        "fit_sample is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(read_idx(std::declval<const IDXOptions&>())), "read_idx is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(idx_element_bytes(std::declval<IDXElement>())),
        "idx_element_bytes is noexcept");
    static_assert(
        noexcept(std::declval<const IDXHeader&>().header_bytes()),
        "header_bytes is noexcept");
    static_assert(
        noexcept(read_idx_header(std::declval<const std::filesystem::path&>())),
        "read_idx_header is noexcept");
    static_assert(
        noexcept(read_idx_payload(std::declval<const std::filesystem::path&>(),
                std::declval<const IDXHeader&>())),
        "read_idx_payload is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(has_image_extension(std::declval<const std::filesystem::path&>())),
        "has_image_extension is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(read_image_folder(std::declval<const ImageFolderOptions&>())),
        "read_image_folder is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(load_sample_image(std::declval<const SampleList&>(), std::declval<std::size_t>(),
                std::declval<const Codecs::DecodeOptions&>(), std::declval<Imaging::Image&>())),
        "load_sample_image is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(load_sample_mask(std::declval<const SampleList&>(), std::declval<std::size_t>(),
                std::declval<Imaging::Image&>())),
        "load_sample_mask is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(read_mask_png(std::declval<const MaskPNGOptions&>())),
        "read_mask_png is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(mnist_available(std::declval<const MNISTOptions&>())),
        "mnist_available is noexcept");
    static_assert(
        noexcept(mnist_expected_paths(std::declval<const MNISTOptions&>())),
        "mnist_expected_paths is noexcept");
    static_assert(
        noexcept(try_load_mnist(std::declval<SushiBLAS::Engine&>(),
                std::declval<const MNISTOptions&>())),
        "try_load_mnist is noexcept");
    static_assert(
        noexcept(make_mnist(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_mnist is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<MultiLabelCollator&>().make(std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(pipeline_table()), "pipeline_table is noexcept");
    static_assert(
        noexcept(find_pipeline(std::declval<std::string_view>())),
        "find_pipeline is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(paint_polygon_mask(std::declval<const Imaging::Labels&>(),
                std::declval<Imaging::ImageSize>(), std::declval<std::uint8_t>(),
                std::declval<Imaging::Image&>())),
        "paint_polygon_mask is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<ReaderDataset&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SampleList>(), std::declval<std::unique_ptr<const Collator>>(),
                std::declval<ReaderDatasetOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<ReaderDataset&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SampleList>(), std::declval<std::unique_ptr<const Collator>>(),
                std::declval<std::unique_ptr<const SamplePipeline>>(),
                std::declval<ReaderDatasetOptions>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(make_reader_dataset(std::declval<SushiBLAS::Engine&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_reader_dataset is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(reader_table()), "reader_table is noexcept");
    static_assert(
        noexcept(find_reader(std::declval<std::string_view>())),
        "find_reader is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<RegressionCollator&>().make(std::declval<std::int32_t>(),
                std::declval<std::int32_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(sample_cache_path(std::declval<const std::filesystem::path&>(),
                std::declval<const SampleCacheOptions&>())),
        "sample_cache_path is noexcept");
    static_assert(
        noexcept(write_sample_cache(std::declval<const SampleList&>(),
                std::declval<const SampleCacheOptions&>())),
        "write_sample_cache is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<SegmentationCollator&>().make(
                std::declval<const SegmentationOptions&>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<const StoredPixels&>().byte_count()),
        "byte_count is noexcept");
    static_assert(
        noexcept(read_stored_pixels(std::declval<std::istream&>(),
                std::declval<const StoredPixels&>(), std::declval<Imaging::Image&>())),
        "read_stored_pixels is noexcept");
    static_assert(
        noexcept(read_stored_pixels(std::declval<const std::filesystem::path&>(),
                std::declval<const StoredPixels&>(), std::declval<Imaging::Image&>())),
        "read_stored_pixels is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<TableDataset&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<SampleList>(), std::declval<std::unique_ptr<const Collator>>(),
                std::declval<TableDatasetOptions>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(task_table()), "task_table is noexcept");
    static_assert(noexcept(find_task(std::declval<std::string_view>())), "find_task is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(noexcept(std::declval<const ValueTable&>().row_width()), "row_width is noexcept");
    static_assert(noexcept(std::declval<const ValueTable&>().row_count()), "row_count is noexcept");
    static_assert(
        noexcept(std::declval<const ValueTable&>().features(std::declval<std::size_t>())),
        "features is noexcept");
    static_assert(
        noexcept(std::declval<const ValueTable&>().targets(std::declval<std::size_t>())),
        "targets is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(read_voc_xml(std::declval<const VOCXMLOptions&>())),
        "read_voc_xml is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(std::declval<const XMLElement&>().find_child(std::declval<std::string_view>())),
        "find_child is noexcept");
    static_assert(noexcept(parse_xml(std::declval<std::string_view>())), "parse_xml is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Datasets
{
    static_assert(
        noexcept(yolo_label_path(std::declval<const std::filesystem::path&>())),
        "yolo_label_path is noexcept");
    static_assert(
        noexcept(read_yolo_txt(std::declval<const YOLOTxtOptions&>())),
        "read_yolo_txt is noexcept");
} // namespace SushiAI::Vision::Datasets

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(cluster_anchors(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<const ClusteringOptions&>())),
        "cluster_anchors is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(choose_anchor_count(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<std::int64_t>(), std::declval<const CountOptions&>())),
        "choose_anchor_count is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(measure_anchor_fit(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<const AnchorLayout&>(), std::declval<const FitOptions&>())),
        "measure_anchor_fit is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(validate_anchor_layout(std::declval<const AnchorLayout&>())),
        "validate_anchor_layout is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(plan_anchors(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<SushiRuntime::span<const std::int64_t>>(),
                std::declval<const AnchorPlanOptions&>())),
        "plan_anchors is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(review_anchors(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<const AnchorLayout&>(),
                std::declval<SushiRuntime::span<const std::int64_t>>(),
                std::declval<const ReviewOptions&>())),
        "review_anchors is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(shape_iou(std::declval<AnchorSize>(), std::declval<AnchorSize>())),
        "shape_iou is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(read_anchor_task(std::declval<const Config::Value&>())),
        "read_anchor_task is noexcept");
    static_assert(
        noexcept(write_anchor_task(std::declval<const AnchorTask&>())),
        "write_anchor_task is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(read_assignment_section(std::declval<const Config::Value*>(),
                std::declval<std::string_view>())),
        "read_assignment_section is noexcept");
    static_assert(
        noexcept(writes_assignment_section(std::declval<const AssignmentChoice&>())),
        "writes_assignment_section is noexcept");
    static_assert(
        noexcept(write_assignment_section(std::declval<const AssignmentChoice&>(),
                std::declval<std::string_view>())),
        "write_assignment_section is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(derive_assignment_thresholds(std::declval<const AnchorFit&>())),
        "derive_assignment_thresholds is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(assign_anchors_to_heads(std::declval<SushiRuntime::span<const AnchorSize>>(),
                std::declval<SushiRuntime::span<const std::int64_t>>(),
                std::declval<SushiRuntime::span<const std::int64_t>>())),
        "assign_anchors_to_heads is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(measure_head_cost(std::declval<const HeadDelta&>(),
                std::declval<const HeadShape&>(), std::declval<const HeadOptions&>(),
                std::declval<const HeadTrainingOptions&>())),
        "measure_head_cost is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(decode_head(std::declval<SushiRuntime::span<const float>>(),
                std::declval<const HeadShape&>(), std::declval<const HeadOptions&>(),
                std::declval<const AnchorLayout&>(), std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "decode_head is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(compute_head_delta(std::declval<SushiRuntime::span<const float>>(),
                std::declval<const HeadShape&>(), std::declval<const HeadOptions&>(),
                std::declval<const AnchorLayout&>(), std::declval<std::int64_t>(),
                std::declval<const HeadTrainingOptions&>(),
                std::declval<const Detect::GroundTruthBatch&>(), std::declval<NetworkSize>(),
                std::declval<HeadDelta&>())),
        "compute_head_delta is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(validate_head(std::declval<const HeadOptions&>(),
                std::declval<const AnchorLayout&>(), std::declval<std::int64_t>())),
        "validate_head is noexcept");
    static_assert(
        noexcept(count_head_channels(std::declval<const HeadOptions&>(),
                std::declval<const AnchorLayout&>(), std::declval<std::int64_t>())),
        "count_head_channels is noexcept");
    static_assert(
        noexcept(count_head_rows(std::declval<const HeadShape&>(),
                std::declval<const AnchorLayout&>(), std::declval<std::int64_t>())),
        "count_head_rows is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(validate_head_training(std::declval<const HeadTrainingOptions&>())),
        "validate_head_training is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(read_head_training_section(std::declval<const Config::Value&>())),
        "read_head_training_section is noexcept");
    static_assert(
        noexcept(write_head_training_section(std::declval<const HeadTrainingOptions&>())),
        "write_head_training_section is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(validate_head_truth(std::declval<const Detect::GroundTruthBatch&>(),
                std::declval<std::int64_t>(), std::declval<std::int64_t>(),
                std::declval<NetworkSize>())),
        "validate_head_truth is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(validate_shape_ratio_options(std::declval<const ShapeRatioOptions&>())),
        "validate_shape_ratio_options is noexcept");
    static_assert(
        noexcept(ShapeRatioAssignment(std::declval<const ShapeRatioOptions&>())),
        "ShapeRatioAssignment is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect::Anchor
{
    static_assert(
        noexcept(objectness_balance(std::declval<std::int64_t>(), std::declval<std::int64_t>())),
        "objectness_balance is noexcept");
    static_assert(
        noexcept(derive_shape_ratio_factors(std::declval<const ShapeRatioDeltaInput&>())),
        "derive_shape_ratio_factors is noexcept");
    static_assert(
        noexcept(validate_shape_ratio_delta(std::declval<const ShapeRatioDeltaInput&>())),
        "validate_shape_ratio_delta is noexcept");
    static_assert(
        noexcept(build_objectness_targets(std::declval<const ShapeRatioDeltaInput&>(),
                std::declval<std::vector<float>&>())),
        "build_objectness_targets is noexcept");
    static_assert(
        noexcept(measure_shape_ratio_loss(std::declval<const ShapeRatioDeltaInput&>(),
                std::declval<SushiRuntime::span<const float>>())),
        "measure_shape_ratio_loss is noexcept");
    static_assert(
        noexcept(compute_shape_ratio_delta(std::declval<const ShapeRatioDeltaInput&>(),
                std::declval<HeadDelta&>())),
        "compute_shape_ratio_delta is noexcept");
} // namespace SushiAI::Vision::Detect::Anchor

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(match_detections(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const std::int64_t>>(),
                std::declval<SushiRuntime::span<std::uint8_t>>())),
        "match_detections is noexcept");
    static_assert(
        noexcept(integrate_precision(std::declval<SushiRuntime::span<const double>>(),
                std::declval<SushiRuntime::span<const double>>())),
        "integrate_precision is noexcept");
    static_assert(
        noexcept(std::declval<AveragePrecision&>().add_image(
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const std::int64_t>>())),
        "add_image is noexcept");
    static_assert(
        noexcept(std::declval<const AveragePrecision&>().summarize()),
        "summarize is noexcept");
    static_assert(noexcept(std::declval<AveragePrecision&>().clear()), "clear is noexcept");
    static_assert(
        noexcept(std::declval<const AveragePrecision&>().image_count()),
        "image_count is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(add_batch(std::declval<AveragePrecision&>(),
                std::declval<const std::vector<std::vector<float>>&>(),
                std::declval<const GroundTruthBatch&>())),
        "add_batch is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(xyxy_to_xywh(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "xyxy_to_xywh is noexcept");
    static_assert(
        noexcept(xywh_to_xyxy(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "xywh_to_xyxy is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(suppress_overlaps(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(), std::declval<float>(),
                std::declval<std::vector<std::int64_t>&>())),
        "suppress_overlaps is noexcept");
    static_assert(
        noexcept(suppress_overlaps(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(), std::declval<float>(),
                std::declval<SuppressionCriterion>(), std::declval<float>(),
                std::declval<std::vector<std::int64_t>&>())),
        "suppress_overlaps is noexcept");
    static_assert(
        noexcept(select_nms_detections(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<const PredictionShape&>(), std::declval<const NMSConfig&>(),
                std::declval<std::vector<std::vector<float>>&>())),
        "select_nms_detections is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(count_top_detections(std::declval<const PredictionShape&>(),
                std::declval<std::int64_t>())),
        "count_top_detections is noexcept");
    static_assert(
        noexcept(select_top_detections(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<const PredictionShape&>(), std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<float>>())),
        "select_top_detections is noexcept");
    static_assert(
        noexcept(keep_confident_detections(std::declval<SushiRuntime::span<const float>>(),
                std::declval<std::int64_t>(), std::declval<float>(),
                std::declval<std::vector<std::vector<float>>&>())),
        "keep_confident_detections is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(first_non_finite(std::declval<SushiRuntime::span<const float>>())),
        "first_non_finite is noexcept");
    static_assert(
        noexcept(ranks_above(std::declval<float>(), std::declval<float>())),
        "ranks_above is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(build_anchor_leaves(std::declval<SushiRuntime::span<const FeatureLevel>>(),
                std::declval<std::int64_t>(), std::declval<AnchorLeaves&>())),
        "build_anchor_leaves is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(count_anchors(std::declval<SushiRuntime::span<const FeatureLevel>>())),
        "count_anchors is noexcept");
    static_assert(
        noexcept(make_anchors(std::declval<SushiRuntime::span<const FeatureLevel>>(),
                std::declval<SushiRuntime::span<float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "make_anchors is noexcept");
    static_assert(
        noexcept(scale_points(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "scale_points is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(std::declval<TaskAlignedAssigner&>().make(std::declval<const AssignerConfig&>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<TaskAlignedAssigner&>().assign(std::declval<const AssignerInputs&>(),
                std::declval<Assignment&>())),
        "assign is noexcept");
    static_assert(
        noexcept(std::declval<const TaskAlignedAssigner&>().config()),
        "config is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(one_to_many_assigner_config(std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<const float>>())),
        "one_to_many_assigner_config is noexcept");
    static_assert(
        noexcept(one_to_one_assigner_config(std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<const float>>())),
        "one_to_one_assigner_config is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(decode_xyxy(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "decode_xyxy is noexcept");
    static_assert(
        noexcept(decode_xywh(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "decode_xywh is noexcept");
    static_assert(
        noexcept(encode_distances(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "encode_distances is noexcept");
    static_assert(
        noexcept(scale_boxes(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "scale_boxes is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(expect_distances(std::declval<SushiRuntime::span<const float>>(),
                std::declval<std::int64_t>(), std::declval<SushiRuntime::span<float>>())),
        "expect_distances is noexcept");
    static_assert(
        noexcept(fill_distribution_projection(std::declval<SushiRuntime::span<float>>())),
        "fill_distribution_projection is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(fill_distribution_weights(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(), std::declval<std::int64_t>(),
                std::declval<SushiRuntime::span<float>>())),
        "fill_distribution_weights is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(read_free_task(std::declval<const Config::Value&>())),
        "read_free_task is noexcept");
    static_assert(
        noexcept(write_free_task(std::declval<const FreeTask&>())),
        "write_free_task is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(detect_port_name(std::declval<std::string_view>(), std::declval<std::size_t>(),
                std::declval<std::string_view>())),
        "detect_port_name is noexcept");
    static_assert(
        noexcept(detect_class_bias(std::declval<std::int64_t>(), std::declval<double>())),
        "detect_class_bias is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(join_level(std::declval<SushiRuntime::span<const float>>(),
                std::declval<const LevelPlacement&>(), std::declval<SushiRuntime::span<float>>())),
        "join_level is noexcept");
    static_assert(
        noexcept(split_level(std::declval<SushiRuntime::span<const float>>(),
                std::declval<const LevelPlacement&>(), std::declval<SushiRuntime::span<float>>())),
        "split_level is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect::Free
{
    static_assert(
        noexcept(build_target_leaves(std::declval<const Assignment&>(),
                std::declval<const TargetLeafInputs&>(), std::declval<TargetLeaves&>())),
        "build_target_leaves is noexcept");
} // namespace SushiAI::Vision::Detect::Free

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(pad_ground_truth(std::declval<const GroundTruthRows&>(),
                std::declval<std::int64_t>(), std::declval<float>(), std::declval<float>(),
                std::declval<GroundTruthBatch&>())),
        "pad_ground_truth is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(intersection_over_union(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>())),
        "intersection_over_union is noexcept");
    static_assert(
        noexcept(complete_iou(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>())),
        "complete_iou is noexcept");
    static_assert(
        noexcept(elementwise_iou(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "elementwise_iou is noexcept");
    static_assert(
        noexcept(elementwise_complete_iou(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "elementwise_complete_iou is noexcept");
    static_assert(
        noexcept(pairwise_iou(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<float>>())),
        "pairwise_iou is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(complete_iou_gradient(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(), std::declval<BoxGradient>(),
                std::declval<SushiRuntime::span<float>>())),
        "complete_iou_gradient is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(measure_suppression(std::declval<SushiRuntime::span<const float>>(),
                std::declval<SushiRuntime::span<const float>>(),
                std::declval<SuppressionCriterion>(), std::declval<float>())),
        "measure_suppression is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Detect
{
    static_assert(
        noexcept(select_top_k(std::declval<SushiRuntime::span<const float>>(),
                std::declval<std::int64_t>(), std::declval<std::vector<std::int64_t>&>())),
        "select_top_k is noexcept");
} // namespace SushiAI::Vision::Detect

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(validate_affine(std::declval<const AffineSettings&>())),
        "validate_affine is noexcept");
    static_assert(
        noexcept(plan_affine(std::declval<const AffineSettings&>(), std::declval<ImageSize>(),
                std::declval<ImageSize>(), std::declval<Random&>())),
        "plan_affine is noexcept");
    static_assert(
        noexcept(apply_affine(std::declval<ImageView>(), std::declval<const AffinePlan&>(),
                std::declval<Image&>())),
        "apply_affine is noexcept");
    static_assert(
        noexcept(apply_affine(std::declval<const Labels&>(), std::declval<const AffinePlan&>())),
        "apply_affine is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(apply_affine_nearest(std::declval<ImageView>(), std::declval<const AffinePlan&>(),
                std::declval<Image&>())),
        "apply_affine_nearest is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(noexcept(make_polygon_filter()), "make_polygon_filter is noexcept");
    static_assert(
        noexcept(to_pixel_box(std::declval<const NormalizedBox&>(), std::declval<ImageSize>())),
        "to_pixel_box is noexcept");
    static_assert(
        noexcept(to_normalized_box(std::declval<const PixelBox&>(), std::declval<ImageSize>())),
        "to_normalized_box is noexcept");
    static_assert(
        noexcept(scale_box(std::declval<const PixelBox&>(), std::declval<float>(),
                std::declval<float>())),
        "scale_box is noexcept");
    static_assert(
        noexcept(translate_box(std::declval<const PixelBox&>(), std::declval<float>(),
                std::declval<float>())),
        "translate_box is noexcept");
    static_assert(
        noexcept(clip_box(std::declval<const PixelBox&>(), std::declval<ImageSize>())),
        "clip_box is noexcept");
    static_assert(noexcept(box_area(std::declval<const PixelBox&>())), "box_area is noexcept");
    static_assert(
        noexcept(is_box_candidate(std::declval<const PixelBox&>(), std::declval<const PixelBox&>(),
                std::declval<const CandidateFilter&>())),
        "is_box_candidate is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(std::declval<DatasetPartners&>().make(std::declval<std::size_t>())),
        "make is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(validate_flip(std::declval<const FlipSettings&>())),
        "validate_flip is noexcept");
    static_assert(
        noexcept(plan_flip(std::declval<const FlipSettings&>(), std::declval<ImageSize>(),
                std::declval<Random&>())),
        "plan_flip is noexcept");
    static_assert(
        noexcept(apply_flip(std::declval<ImageView>(), std::declval<const FlipPlan&>(),
                std::declval<Image&>())),
        "apply_flip is noexcept");
    static_assert(
        noexcept(apply_flip(std::declval<const Labels&>(), std::declval<const FlipPlan&>())),
        "apply_flip is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(expand_gray(std::declval<ImageView>(), std::declval<Image&>())),
        "expand_gray is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(validate_hsv(std::declval<const HSVSettings&>())),
        "validate_hsv is noexcept");
    static_assert(
        noexcept(plan_hsv(std::declval<const HSVSettings&>(), std::declval<Random&>())),
        "plan_hsv is noexcept");
    static_assert(
        noexcept(is_hsv_identity(std::declval<const HSVPlan&>())),
        "is_hsv_identity is noexcept");
    static_assert(
        noexcept(make_hsv_tables(std::declval<const HSVPlan&>())),
        "make_hsv_tables is noexcept");
    static_assert(noexcept(to_hsv(std::declval<RGBPixel>())), "to_hsv is noexcept");
    static_assert(noexcept(to_rgb(std::declval<HSVPixel>())), "to_rgb is noexcept");
    static_assert(
        noexcept(apply_hsv(std::declval<ImageView>(), std::declval<const HSVPlan&>(),
                std::declval<Image&>())),
        "apply_hsv is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(is_valid_side(std::declval<std::int64_t>())),
        "is_valid_side is noexcept");
    static_assert(
        noexcept(is_valid_channel_count(std::declval<std::int64_t>())),
        "is_valid_channel_count is noexcept");
    static_assert(noexcept(is_valid_size(std::declval<ImageSize>())), "is_valid_size is noexcept");
    static_assert(
        noexcept(ImageView(std::declval<const std::uint8_t*>(), std::declval<ImageSize>(),
                std::declval<std::int32_t>())),
        "ImageView is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().data()), "data is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().size()), "size is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().height()), "height is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().width()), "width is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().channels()), "channels is noexcept");
    static_assert(noexcept(std::declval<const ImageView&>().row_bytes()), "row_bytes is noexcept");
    static_assert(
        noexcept(std::declval<const ImageView&>().byte_count()),
        "byte_count is noexcept");
    static_assert(
        noexcept(std::declval<const ImageView&>().row(std::declval<std::int32_t>())),
        "row is noexcept");
    static_assert(
        noexcept(Image(std::declval<ImageSize>(), std::declval<std::int32_t>())),
        "Image is noexcept");
    static_assert(
        noexcept(std::declval<Image&>().reset(std::declval<ImageSize>(),
                std::declval<std::int32_t>())),
        "reset is noexcept");
    static_assert(
        noexcept(std::declval<Image&>().fill(std::declval<std::uint8_t>())),
        "fill is noexcept");
    static_assert(noexcept(std::declval<Image&>().data()), "data is noexcept");
    static_assert(noexcept(std::declval<const Image&>().data()), "data is noexcept");
    static_assert(
        noexcept(std::declval<const Image&>().allocated_bytes()),
        "allocated_bytes is noexcept");
    static_assert(noexcept(std::declval<const Image&>().size()), "size is noexcept");
    static_assert(noexcept(std::declval<const Image&>().height()), "height is noexcept");
    static_assert(noexcept(std::declval<const Image&>().width()), "width is noexcept");
    static_assert(noexcept(std::declval<const Image&>().channels()), "channels is noexcept");
    static_assert(noexcept(std::declval<const Image&>().row_bytes()), "row_bytes is noexcept");
    static_assert(noexcept(std::declval<const Image&>().byte_count()), "byte_count is noexcept");
    static_assert(
        noexcept(std::declval<Image&>().row(std::declval<std::int32_t>())),
        "row is noexcept");
    static_assert(
        noexcept(std::declval<const Image&>().row(std::declval<std::int32_t>())),
        "row is noexcept");
    static_assert(noexcept(std::declval<const Image&>().view()), "view is noexcept");
    static_assert(
        noexcept(overlaps(std::declval<ImageView>(), std::declval<const Image&>())),
        "overlaps is noexcept");
    static_assert(
        noexcept(check_source(std::declval<ImageView>(), std::declval<const Image&>(),
                std::declval<std::string_view>())),
        "check_source is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging::SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(clip_labels(std::declval<Labels&>(), std::declval<ImageSize>())),
        "clip_labels is noexcept");
} // namespace SushiAI::Vision::Imaging::SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(parse_label_text(std::declval<std::string_view>(), std::declval<std::int32_t>())),
        "parse_label_text is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(validate_letterbox(std::declval<const LetterboxSettings&>())),
        "validate_letterbox is noexcept");
    static_assert(
        noexcept(plan_letterbox(std::declval<const LetterboxSettings&>(), std::declval<ImageSize>(),
                std::declval<ImageSize>())),
        "plan_letterbox is noexcept");
    static_assert(
        noexcept(apply_letterbox(std::declval<ImageView>(), std::declval<const LetterboxPlan&>(),
                std::declval<Image&>())),
        "apply_letterbox is noexcept");
    static_assert(
        noexcept(apply_letterbox(std::declval<const Labels&>(),
                std::declval<const LetterboxPlan&>())),
        "apply_letterbox is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(TrainingMaskPipeline(std::declval<const MaskSource&>(),
                std::declval<const TrainingSettings&>(), std::declval<std::uint8_t>())),
        "TrainingMaskPipeline is noexcept");
    static_assert(
        noexcept(std::declval<TrainingMaskPipeline&>().make_mask(std::declval<std::size_t>(),
                std::declval<const TrainingTrace&>(), std::declval<ImageSize>(),
                std::declval<Image&>())),
        "make_mask is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(validate_mosaic(std::declval<const MosaicSettings&>())),
        "validate_mosaic is noexcept");
    static_assert(
        noexcept(validate_mosaic_size(std::declval<std::int32_t>())),
        "validate_mosaic_size is noexcept");
    static_assert(
        noexcept(draw_mosaic(std::declval<const MosaicSettings&>(), std::declval<std::int32_t>(),
                std::declval<std::size_t>(), std::declval<PartnerSource&>(),
                std::declval<Random&>())),
        "draw_mosaic is noexcept");
    static_assert(
        noexcept(plan_mosaic(std::declval<const MosaicSettings&>(),
                std::declval<const MosaicDraw&>(), std::declval<std::int32_t>(),
                std::declval<const std::array<ImageSize, K_MOSAIC_PATCHES>&>())),
        "plan_mosaic is noexcept");
    static_assert(
        noexcept(apply_mosaic(std::declval<const std::array<ImageView, K_MOSAIC_PATCHES>&>(),
                std::declval<const MosaicPlan&>(), std::declval<Image&>())),
        "apply_mosaic is noexcept");
    static_assert(
        noexcept(apply_mosaic(std::declval<const std::array<Labels, K_MOSAIC_PATCHES>&>(),
                std::declval<const MosaicPlan&>())),
        "apply_mosaic is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(is_mosaic_closed(std::declval<std::int32_t>(), std::declval<std::int32_t>(),
                std::declval<std::int32_t>())),
        "is_mosaic_closed is noexcept");
    static_assert(
        noexcept(validate_training(std::declval<const TrainingSettings&>())),
        "validate_training is noexcept");
    static_assert(
        noexcept(make_sample_random(std::declval<std::uint64_t>(), std::declval<std::int32_t>(),
                std::declval<std::size_t>())),
        "make_sample_random is noexcept");
    static_assert(
        noexcept(std::declval<TrainingPipeline&>().make(std::declval<const SampleSource&>(),
                std::declval<const TrainingSettings&>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<TrainingPipeline&>().make(std::declval<const SampleSource&>(),
                std::declval<const TrainingSettings&>(),
                std::declval<std::unique_ptr<PartnerSource>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<TrainingPipeline&>().make_sample(std::declval<std::size_t>(),
                std::declval<std::int32_t>(), std::declval<Sample&>())),
        "make_sample is noexcept");
    static_assert(
        noexcept(std::declval<const TrainingPipeline&>().settings()),
        "settings is noexcept");
    static_assert(
        noexcept(validate_validation(std::declval<const ValidationSettings&>())),
        "validate_validation is noexcept");
    static_assert(
        noexcept(std::declval<ValidationPipeline&>().make(std::declval<const SampleSource&>(),
                std::declval<const ValidationSettings&>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<ValidationPipeline&>().make_sample(std::declval<std::size_t>(),
                std::declval<ImageSize>(), std::declval<Sample&>())),
        "make_sample is noexcept");
    static_assert(
        noexcept(std::declval<const ValidationPipeline&>().settings()),
        "settings is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(resampled_point_count(std::declval<std::size_t>())),
        "resampled_point_count is noexcept");
    static_assert(
        noexcept(resample_polygon(std::declval<const Polygon&>(), std::declval<std::size_t>())),
        "resample_polygon is noexcept");
    static_assert(
        noexcept(map_polygon(std::declval<Polygon&>(), std::declval<float>(), std::declval<float>(),
                std::declval<float>(), std::declval<float>())),
        "map_polygon is noexcept");
    static_assert(
        noexcept(clamp_polygon(std::declval<Polygon&>(), std::declval<const PixelBox&>())),
        "clamp_polygon is noexcept");
    static_assert(
        noexcept(is_point_inside(std::declval<const Polygon&>(), std::declval<Point>())),
        "is_point_inside is noexcept");
    static_assert(
        noexcept(visible_polygon_box(std::declval<const Polygon&>(), std::declval<ImageSize>())),
        "visible_polygon_box is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(noexcept(Random(std::declval<std::uint64_t>())), "Random is noexcept");
    static_assert(
        noexcept(std::declval<Random&>().mix(std::declval<std::uint64_t>())),
        "mix is noexcept");
    static_assert(
        noexcept(std::declval<const Random&>().derive(std::declval<std::uint64_t>())),
        "derive is noexcept");
    static_assert(noexcept(std::declval<Random&>().next_u64()), "next_u64 is noexcept");
    static_assert(noexcept(std::declval<Random&>().next_u32()), "next_u32 is noexcept");
    static_assert(noexcept(std::declval<Random&>().uniform()), "uniform is noexcept");
    static_assert(
        noexcept(std::declval<Random&>().uniform(std::declval<float>(), std::declval<float>())),
        "uniform is noexcept");
    static_assert(
        noexcept(std::declval<Random&>().index(std::declval<std::uint32_t>())),
        "index is noexcept");
    static_assert(
        noexcept(std::declval<Random&>().bernoulli(std::declval<float>())),
        "bernoulli is noexcept");
    static_assert(noexcept(std::declval<const Random&>().state()), "state is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(std::declval<const RawImageHeader&>().pixel_bytes()),
        "pixel_bytes is noexcept");
    static_assert(
        noexcept(write_raw_image(std::declval<std::ostream&>(), std::declval<ImageView>(),
                std::declval<ChannelOrder>())),
        "write_raw_image is noexcept");
    static_assert(
        noexcept(read_raw_image_header(std::declval<std::istream&>())),
        "read_raw_image_header is noexcept");
    static_assert(
        noexcept(read_raw_image_pixels(std::declval<std::istream&>(),
                std::declval<const RawImageHeader&>(), std::declval<Image&>())),
        "read_raw_image_pixels is noexcept");
    static_assert(
        noexcept(write_raw_image_file(std::declval<const std::filesystem::path&>(),
                std::declval<ImageView>(), std::declval<ChannelOrder>())),
        "write_raw_image_file is noexcept");
    static_assert(
        noexcept(read_raw_image_file(std::declval<const std::filesystem::path&>(),
                std::declval<Image&>())),
        "read_raw_image_file is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(recent_capacity(std::declval<std::size_t>(), std::declval<std::int32_t>())),
        "recent_capacity is noexcept");
    static_assert(
        noexcept(std::declval<RecentPartners&>().make(std::declval<std::size_t>(),
                std::declval<std::int32_t>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const RecentPartners&>().recent()), "recent is noexcept");
    static_assert(
        noexcept(std::declval<const RecentPartners&>().capacity()),
        "capacity is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(std::declval<const RectangularBatches&>().shape_at(std::declval<std::size_t>())),
        "shape_at is noexcept");
    static_assert(
        noexcept(validate_rectangle(std::declval<const RectangleSettings&>())),
        "validate_rectangle is noexcept");
    static_assert(
        noexcept(plan_rectangular_batches(std::declval<const std::vector<ImageSize>&>(),
                std::declval<const RectangleSettings&>())),
        "plan_rectangular_batches is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Imaging
{
    static_assert(
        noexcept(fit_long_side(std::declval<ImageSize>(), std::declval<std::int32_t>())),
        "fit_long_side is noexcept");
    static_assert(
        noexcept(resize_bilinear(std::declval<ImageView>(), std::declval<ImageSize>(),
                std::declval<Image&>())),
        "resize_bilinear is noexcept");
    static_assert(
        noexcept(resize_nearest(std::declval<ImageView>(), std::declval<ImageSize>(),
                std::declval<Image&>())),
        "resize_nearest is noexcept");
    static_assert(
        noexcept(resize_long_side(std::declval<ImageView>(), std::declval<std::int32_t>(),
                std::declval<Image&>())),
        "resize_long_side is noexcept");
} // namespace SushiAI::Vision::Imaging

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(parse_darknet_config(std::declval<std::string_view>(),
                std::declval<std::string_view>())),
        "parse_darknet_config is noexcept");
    static_assert(
        noexcept(load_darknet_config(std::declval<const std::filesystem::path&>())),
        "load_darknet_config is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(fold_darknet_normalisation(std::declval<DarknetConvolution&>())),
        "fold_darknet_normalisation is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(import_darknet_model(std::declval<const DarknetConfig&>(),
                std::declval<const DarknetModel&>(), std::declval<const DarknetWeights*>())),
        "import_darknet_model is noexcept");
    static_assert(
        noexcept(import_darknet_files(std::declval<SushiRuntime::span<const std::filesystem::path>>(
                ))),
        "import_darknet_files is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(translate_darknet_model(std::declval<const DarknetConfig&>(),
                std::declval<const DarknetTranslation&>())),
        "translate_darknet_model is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(write_darknet_recipe(std::declval<const DarknetConfig&>(),
                std::declval<const DarknetTraining&>())),
        "write_darknet_recipe is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(store_darknet_weights(std::declval<const DarknetWeights&>(),
                std::declval<SushiRuntime::span<const DarknetConvolutionShape>>(),
                std::declval<SushiRuntime::span<const DarknetWeightTarget>>(),
                std::declval<NN::ModelState&>())),
        "store_darknet_weights is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(list_darknet_tensors(std::declval<const DarknetWeights&>(),
                std::declval<SushiRuntime::span<const DarknetConvolutionShape>>(),
                std::declval<SushiRuntime::span<const DarknetWeightTarget>>())),
        "list_darknet_tensors is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(read_darknet_training(std::declval<const DarknetConfig&>())),
        "read_darknet_training is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision::Importers
{
    static_assert(
        noexcept(read_darknet_header(std::declval<std::istream&>())),
        "read_darknet_header is noexcept");
    static_assert(
        noexcept(read_darknet_weights(std::declval<const std::filesystem::path&>(),
                std::declval<SushiRuntime::span<const DarknetConvolutionShape>>())),
        "read_darknet_weights is noexcept");
} // namespace SushiAI::Vision::Importers

namespace SushiAI::Vision
{
    static_assert(
        noexcept(register_nodes(std::declval<NN::NodeRegistry&>())),
        "register_nodes is noexcept");
    static_assert(
        noexcept(register_datasets(std::declval<Data::DatasetRegistry&>())),
        "register_datasets is noexcept");
    static_assert(
        noexcept(register_objectives(std::declval<Train::ObjectiveRegistry&>())),
        "register_objectives is noexcept");
    static_assert(
        noexcept(register_importers(std::declval<IO::ImporterRegistry&>())),
        "register_importers is noexcept");
} // namespace SushiAI::Vision

namespace SushiAI::Vision::Segment
{
    static_assert(
        noexcept(std::declval<ConfusionMatrix&>().make(std::declval<std::int32_t>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<ConfusionMatrix&>().add(
                std::declval<SushiRuntime::span<const std::int32_t>>(),
                std::declval<SushiRuntime::span<const std::int32_t>>(), std::declval<std::int32_t>(
                ))),
        "add is noexcept");
    static_assert(
        noexcept(std::declval<ConfusionMatrix&>().merge(std::declval<const ConfusionMatrix&>())),
        "merge is noexcept");
    static_assert(noexcept(std::declval<ConfusionMatrix&>().clear()), "clear is noexcept");
    static_assert(
        noexcept(std::declval<const ConfusionMatrix&>().class_count()),
        "class_count is noexcept");
    static_assert(
        noexcept(std::declval<const ConfusionMatrix&>().count(std::declval<std::int32_t>(),
                std::declval<std::int32_t>())),
        "count is noexcept");
    static_assert(
        noexcept(std::declval<const ConfusionMatrix&>().truth_pixels(std::declval<std::int32_t>())),
        "truth_pixels is noexcept");
    static_assert(
        noexcept(std::declval<const ConfusionMatrix&>().predicted_pixels(std::declval<std::int32_t>(
                ))),
        "predicted_pixels is noexcept");
    static_assert(
        noexcept(std::declval<const ConfusionMatrix&>().pixel_count()),
        "pixel_count is noexcept");
} // namespace SushiAI::Vision::Segment

namespace SushiAI::Vision::Segment
{
    static_assert(
        noexcept(fill_outline(std::declval<SushiRuntime::span<const Imaging::Polygon>>(),
                std::declval<std::uint8_t>(), std::declval<Imaging::Image&>())),
        "fill_outline is noexcept");
} // namespace SushiAI::Vision::Segment

namespace SushiAI::Vision::Segment
{
    static_assert(
        noexcept(class_occurs(std::declval<const ConfusionMatrix&>(),
                std::declval<std::int32_t>())),
        "class_occurs is noexcept");
    static_assert(
        noexcept(class_iou(std::declval<const ConfusionMatrix&>(), std::declval<std::int32_t>())),
        "class_iou is noexcept");
    static_assert(
        noexcept(mean_iou(std::declval<const ConfusionMatrix&>())),
        "mean_iou is noexcept");
} // namespace SushiAI::Vision::Segment

namespace SushiAI::Vision::Segment
{
    static_assert(
        noexcept(pixel_accuracy(std::declval<const ConfusionMatrix&>())),
        "pixel_accuracy is noexcept");
} // namespace SushiAI::Vision::Segment

namespace SushiAI::Vision::Segment
{
    static_assert(
        noexcept(rasterise_outlines(std::declval<SushiRuntime::span<const ClassOutline>>(),
                std::declval<Imaging::ImageSize>(), std::declval<std::uint8_t>(),
                std::declval<Imaging::Image&>())),
        "rasterise_outlines is noexcept");
} // namespace SushiAI::Vision::Segment

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(anchor_objectness_bias(std::declval<std::int64_t>())),
        "anchor_objectness_bias is noexcept");
    static_assert(
        noexcept(anchor_class_bias(std::declval<std::int64_t>())),
        "anchor_class_bias is noexcept");
    static_assert(
        noexcept(derive_anchor_bias_heads(std::declval<const Config::ModelSpec&>(),
                std::declval<const Shape&>(), std::declval<const Detect::Anchor::AnchorTask&>(),
                std::declval<const std::vector<Train::OutputShape>&>())),
        "derive_anchor_bias_heads is noexcept");
    static_assert(
        noexcept(initialize_anchor_biases(std::declval<const Detect::Anchor::AnchorTask&>(),
                std::declval<const std::vector<AnchorBiasHead>&>(),
                std::declval<NN::ModelState&>())),
        "initialize_anchor_biases is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<AnchorObjective&>().make(std::declval<Detect::Anchor::NetworkSize>(),
                std::declval<Detect::Anchor::AnchorLayout>(),
                std::declval<std::vector<AnchorHead>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<AnchorObjective&>().make(std::declval<Detect::Anchor::NetworkSize>(),
                std::declval<Detect::Anchor::AnchorLayout>(),
                std::declval<std::vector<AnchorHead>>(),
                std::declval<std::shared_ptr<const HeadDeltaRule>>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorObjective&>().compute_deltas(
                std::declval<SushiRuntime::span<const SushiRuntime::span<const float>>>(),
                std::declval<const Detect::GroundTruthBatch&>(),
                std::declval<SushiRuntime::span<Detect::Anchor::HeadDelta>>())),
        "compute_deltas is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorObjective&>().measure_cost(
                std::declval<SushiRuntime::span<const Detect::Anchor::HeadDelta>>())),
        "measure_cost is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorObjective&>().leaf_role(std::declval<std::size_t>())),
        "leaf_role is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorObjective&>().network()),
        "network is noexcept");
    static_assert(noexcept(std::declval<const AnchorObjective&>().layout()), "layout is noexcept");
    static_assert(noexcept(std::declval<const AnchorObjective&>().heads()), "heads is noexcept");
    static_assert(noexcept(std::declval<const AnchorObjective&>().rule()), "rule is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(build_anchor_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>())),
        "build_anchor_objective is noexcept");
    static_assert(
        noexcept(make_anchor_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_anchor_objective is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(run_anchor_training(std::declval<AnchorTrainingStep&>(),
                std::declval<const Data::Dataset&>(), std::declval<const AnchorLoopOptions&>())),
        "run_anchor_training is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<AnchorTrainingStep&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const AnchorObjective&>(), std::declval<const Train::TrainingGraph&>(),
                std::declval<Graph::TensorPool&>(), std::declval<Optim::Optimizer&>(),
                std::declval<std::vector<Train::BufferUpdateSlot>>(),
                std::declval<Train::TrainingStepOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<AnchorTrainingStep&>().run(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<const Detect::GroundTruthBatch&>())),
        "run is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorTrainingStep&>().deltas()),
        "deltas is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorTrainingStep&>().output(std::declval<std::string_view>(
                ))),
        "output is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorTrainingStep&>().training()),
        "training is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorTrainingStep&>().run_count()),
        "run_count is noexcept");
    static_assert(
        noexcept(std::declval<AnchorTrainingStep&>().restore_run_count(std::declval<std::size_t>(
                ))),
        "restore_run_count is noexcept");
    static_assert(
        noexcept(std::declval<const AnchorTrainingStep&>().objective()),
        "objective is noexcept");
    static_assert(
        noexcept(std::declval<AnchorTrainingStep&>().optimizer()),
        "optimizer is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(check_detection_run(std::declval<const Data::Dataset&>(),
                std::declval<const DetectionRunSubject&>(),
                std::declval<const DetectionRunPlan&>())),
        "check_detection_run is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(check_truth_layout(std::declval<const Detect::GroundTruthBatch&>(),
                std::declval<std::int64_t>(), std::declval<std::string_view>())),
        "check_truth_layout is noexcept");
    static_assert(
        noexcept(check_truth_labels(std::declval<const Detect::GroundTruthBatch&>(),
                std::declval<std::int64_t>(), std::declval<std::string_view>())),
        "check_truth_labels is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<DiceObjective&>().make(std::declval<const PixelExtents&>(),
                std::declval<std::int32_t>(), std::declval<DType>())),
        "make is noexcept");
    static_assert(noexcept(std::declval<const DiceObjective&>().extents()), "extents is noexcept");
    static_assert(
        noexcept(std::declval<const DiceObjective&>().ignore_index()),
        "ignore_index is noexcept");
    static_assert(
        noexcept(make_dice_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_dice_objective is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(fill_dice_targets(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::int64_t>(), std::declval<std::int32_t>(),
                std::declval<const DiceLeaves&>())),
        "fill_dice_targets is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(emit_distribution_expectation(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>())),
        "emit_distribution_expectation is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(emit_distribution_focal_loss(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>())),
        "emit_distribution_focal_loss is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<FreeAssignmentStage&>().make(std::declval<const FreeModelFacts&>(),
                std::declval<const Detect::Free::FreeTask&>(), std::declval<std::int64_t>(),
                std::declval<FreeBranch>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const FreeAssignmentStage&>().leaves()),
        "leaves is noexcept");
    static_assert(
        noexcept(std::declval<FreeAssignmentStage&>().assign(
                std::declval<SushiRuntime::span<const FreeLevelViews>>(),
                std::declval<const Detect::GroundTruthBatch&>())),
        "assign is noexcept");
    static_assert(
        noexcept(std::declval<const FreeAssignmentStage&>().write_leaf(std::declval<std::size_t>(),
                std::declval<SushiRuntime::span<float>>())),
        "write_leaf is noexcept");
    static_assert(
        noexcept(std::declval<const FreeAssignmentStage&>().assignment()),
        "assignment is noexcept");
    static_assert(
        noexcept(std::declval<const FreeAssignmentStage&>().targets()),
        "targets is noexcept");
    static_assert(
        noexcept(std::declval<const FreeAssignmentStage&>().distribution_weights()),
        "distribution_weights is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<FreeBatchDecoder&>().make(std::declval<const FreeModelFacts&>(),
                std::declval<std::int64_t>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const FreeBatchDecoder&>().anchors()),
        "anchors is noexcept");
    static_assert(
        noexcept(std::declval<FreeBatchDecoder&>().decode(
                std::declval<SushiRuntime::span<const FreeLevelLogits>>(),
                std::declval<SushiRuntime::span<float>>(), std::declval<SushiRuntime::span<float>>(
                ))),
        "decode is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(free_class_bias(std::declval<std::int64_t>(), std::declval<std::int64_t>())),
        "free_class_bias is noexcept");
    static_assert(
        noexcept(initialize_free_biases(std::declval<const FreeModelFacts&>(),
                std::declval<std::string_view>(), std::declval<NN::ModelState&>())),
        "initialize_free_biases is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(validate_free_branch_weights(std::declval<const FreeBranchWeights&>())),
        "validate_free_branch_weights is noexcept");
    static_assert(
        noexcept(read_free_branch_weights(std::declval<const Config::Value&>())),
        "read_free_branch_weights is noexcept");
    static_assert(
        noexcept(one_to_many_weight(std::declval<const FreeBranchWeights&>(),
                std::declval<std::int64_t>(), std::declval<std::int64_t>())),
        "one_to_many_weight is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(emit_free_detection_loss(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<const FreeModelFacts&>(),
                std::declval<const FreeLossValues&>(), std::declval<const FreeLossGains&>())),
        "emit_free_detection_loss is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(evaluate_free_detection_loss(
                std::declval<SushiRuntime::span<const FreeLevelLossArrays>>(),
                std::declval<std::int64_t>(), std::declval<std::int64_t>(),
                std::declval<std::int64_t>(), std::declval<double>(),
                std::declval<const FreeLossGains&>())),
        "evaluate_free_detection_loss is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(validate_free_loss_gains(std::declval<const FreeLossGains&>())),
        "validate_free_loss_gains is noexcept");
    static_assert(
        noexcept(read_free_loss_gains(std::declval<const Config::Value&>())),
        "read_free_loss_gains is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(derive_free_model_facts(std::declval<const Shape&>(),
                std::declval<const std::vector<Train::OutputShape>&>())),
        "derive_free_model_facts is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<FreeObjective&>().make(std::declval<FreeModelFacts>(),
                std::declval<Detect::Free::FreeTask>(), std::declval<std::int64_t>(),
                std::declval<FreeLossGains>(), std::declval<FreeBranchWeights>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const FreeObjective&>().reads_leaf(std::declval<FreeLeafKind>())),
        "reads_leaf is noexcept");
    static_assert(
        noexcept(std::declval<const FreeObjective&>().stage_leaves()),
        "stage_leaves is noexcept");
    static_assert(noexcept(std::declval<const FreeObjective&>().facts()), "facts is noexcept");
    static_assert(noexcept(std::declval<const FreeObjective&>().task()), "task is noexcept");
    static_assert(noexcept(std::declval<const FreeObjective&>().batch()), "batch is noexcept");
    static_assert(noexcept(std::declval<const FreeObjective&>().gains()), "gains is noexcept");
    static_assert(
        noexcept(std::declval<const FreeObjective&>().branch_weights()),
        "branch_weights is noexcept");
    static_assert(
        noexcept(add_free_loss_metrics(std::declval<const FreeLossTerms&>(),
                std::declval<Train::Metrics&>())),
        "add_free_loss_metrics is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(build_free_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>())),
        "build_free_objective is noexcept");
    static_assert(
        noexcept(make_free_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_free_objective is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(run_free_training(std::declval<FreeTrainingStep&>(),
                std::declval<const Data::Dataset&>(), std::declval<const FreeLoopOptions&>())),
        "run_free_training is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<FreeTrainingStep&>().make(std::declval<SushiBLAS::Engine&>(),
                std::declval<const FreeObjective&>(), std::declval<const Train::TrainingGraph&>(),
                std::declval<Graph::TensorPool&>(), std::declval<Optim::Optimizer&>(),
                std::declval<std::vector<Train::BufferUpdateSlot>>(),
                std::declval<Train::TrainingStepOptions>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<FreeTrainingStep&>().run(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<const Detect::GroundTruthBatch&>())),
        "run is noexcept");
    static_assert(noexcept(std::declval<const FreeTrainingStep&>().stage()), "stage is noexcept");
    static_assert(
        noexcept(std::declval<FreeTrainingStep&>().set_one_to_many_weight(std::declval<float>())),
        "set_one_to_many_weight is noexcept");
    static_assert(
        noexcept(std::declval<const FreeTrainingStep&>().one_to_many_weight()),
        "one_to_many_weight is noexcept");
    static_assert(
        noexcept(std::declval<const FreeTrainingStep&>().training()),
        "training is noexcept");
    static_assert(
        noexcept(std::declval<const FreeTrainingStep&>().run_count()),
        "run_count is noexcept");
    static_assert(
        noexcept(std::declval<FreeTrainingStep&>().restore_run_count(std::declval<std::size_t>())),
        "restore_run_count is noexcept");
    static_assert(
        noexcept(std::declval<const FreeTrainingStep&>().objective()),
        "objective is noexcept");
    static_assert(noexcept(std::declval<FreeTrainingStep&>().optimizer()), "optimizer is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(emit_head_delta_loss(std::declval<const Graph::GraphBuilder&>(),
                std::declval<Graph::ValueID>(), std::declval<Graph::ValueID>(),
                std::declval<double>())),
        "emit_head_delta_loss is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(choose_head_delta_rule(std::declval<const Detect::Anchor::AssignmentChoice&>())),
        "choose_head_delta_rule is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(std::declval<PixelCrossEntropyObjective&>().make(std::declval<const PixelExtents&>(
                ), std::declval<std::int32_t>(), std::declval<DType>())),
        "make is noexcept");
    static_assert(
        noexcept(std::declval<const PixelCrossEntropyObjective&>().extents()),
        "extents is noexcept");
    static_assert(
        noexcept(std::declval<const PixelCrossEntropyObjective&>().ignore_index()),
        "ignore_index is noexcept");
    static_assert(
        noexcept(make_pixel_cross_entropy_objective(std::declval<const Train::ObjectiveContext&>(),
                std::declval<const Config::Value&>(), std::declval<std::string_view>())),
        "make_pixel_cross_entropy_objective is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(noexcept(std::declval<const PixelExtents&>().pixels()), "pixels is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(validate_pixel_labels(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::int64_t>(), std::declval<std::int64_t>(),
                std::declval<std::int32_t>())),
        "validate_pixel_labels is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(emit_pixel_cross_entropy(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<Graph::ValueID>(), std::declval<Graph::ValueID>())),
        "emit_pixel_cross_entropy is noexcept");
    static_assert(
        noexcept(emit_soft_dice(std::declval<const Graph::GraphBuilder&>(),
                std::declval<const Graph::Graph&>(), std::declval<Graph::ValueID>(),
                std::declval<const DiceOperands&>())),
        "emit_soft_dice is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(fill_pixel_targets(std::declval<const SushiBLAS::Tensor&>(),
                std::declval<std::int64_t>(), std::declval<std::int32_t>(),
                std::declval<SushiBLAS::Tensor&>(), std::declval<SushiBLAS::Tensor&>())),
        "fill_pixel_targets is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(notify_guarded(std::declval<const std::function<void()>&>())),
        "notify_guarded is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(measure_segmentation(std::declval<const Train::StepView&>(),
                std::declval<const PixelExtents&>(), std::declval<std::int32_t>(),
                std::declval<Train::Metrics&>())),
        "measure_segmentation is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(ShapeRatioHeadDeltaRule(std::declval<const Detect::Anchor::ShapeRatioOptions&>())),
        "ShapeRatioHeadDeltaRule is noexcept");
} // namespace SushiAI::Vision::Training

namespace SushiAI::Vision::Training
{
    static_assert(
        noexcept(describe_truth_roles(std::declval<std::int64_t>(), std::declval<std::int64_t>())),
        "describe_truth_roles is noexcept");
    static_assert(
        noexcept(truth_from_roles(std::declval<const Data::Batch&>(),
                std::declval<Detect::GroundTruthBatch&>())),
        "truth_from_roles is noexcept");
} // namespace SushiAI::Vision::Training

/** @brief Passes when the file compiled: the assertions above are the test. */
TEST(Small_NoexceptSurface, EveryPublicFunctionIsNoexcept)
{
    SUCCEED();
}
