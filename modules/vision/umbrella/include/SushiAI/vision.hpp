/**************************************************************************/
/* vision.hpp                                                             */
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
 * @file vision.hpp
 * @brief Includes SushiAI and every public header of the vision domain.
 * @author Mustafa Garip
 */

#pragma once

#include <SushiAI/SushiAI.hpp>

#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/boxes.hpp>
#include <SushiAI/vision/imaging/dataset_partners.hpp>
#include <SushiAI/vision/imaging/flip.hpp>
#include <SushiAI/vision/imaging/gray.hpp>
#include <SushiAI/vision/imaging/hsv.hpp>
#include <SushiAI/vision/imaging/image.hpp>
#include <SushiAI/vision/imaging/label_set.hpp>
#include <SushiAI/vision/imaging/labels.hpp>
#include <SushiAI/vision/imaging/letterbox.hpp>
#include <SushiAI/vision/imaging/mosaic.hpp>
#include <SushiAI/vision/imaging/partner_source.hpp>
#include <SushiAI/vision/imaging/pipeline.hpp>
#include <SushiAI/vision/imaging/polygon.hpp>
#include <SushiAI/vision/imaging/random.hpp>
#include <SushiAI/vision/imaging/raw_cache.hpp>
#include <SushiAI/vision/imaging/recent_partners.hpp>
#include <SushiAI/vision/imaging/rectangle.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

#include <SushiAI/vision/detect/free/anchor_leaves.hpp>
#include <SushiAI/vision/detect/free/anchors.hpp>
#include <SushiAI/vision/detect/free/assigner.hpp>
#include <SushiAI/vision/detect/free/assigner_config.hpp>
#include <SushiAI/vision/detect/free/assignment.hpp>
#include <SushiAI/vision/detect/average_precision.hpp>
#include <SushiAI/vision/detect/free/box_coding.hpp>
#include <SushiAI/vision/detect/box_format.hpp>
#include <SushiAI/vision/detect/detection_row.hpp>
#include <SushiAI/vision/detect/detections_nms.hpp>
#include <SushiAI/vision/detect/detections_topk.hpp>
#include <SushiAI/vision/detect/ground_truth.hpp>
#include <SushiAI/vision/detect/iou.hpp>
#include <SushiAI/vision/detect/free/level_rows.hpp>
#include <SushiAI/vision/detect/prediction_shape.hpp>
#include <SushiAI/vision/detect/free/target_leaves.hpp>
#include <SushiAI/vision/detect/free/distribution_expectation.hpp>
#include <SushiAI/vision/detect/free/distribution_targets.hpp>
#include <SushiAI/vision/detect/free/free_task.hpp>
#include <SushiAI/vision/detect/free/head_contract.hpp>
#include <SushiAI/vision/detect/top_k.hpp>

#include <SushiAI/vision/blocks/attention.hpp>
#include <SushiAI/vision/blocks/bottleneck.hpp>
#include <SushiAI/vision/blocks/c2psa.hpp>
#include <SushiAI/vision/blocks/c3k.hpp>
#include <SushiAI/vision/blocks/c3k2.hpp>
#include <SushiAI/vision/blocks/conv_norm.hpp>
#include <SushiAI/vision/blocks/detect_head.hpp>
#include <SushiAI/vision/blocks/psa_block.hpp>
#include <SushiAI/vision/blocks/sppf.hpp>

#include <SushiAI/vision/datasets/batch_inputs.hpp>
#include <SushiAI/vision/datasets/binary_collator.hpp>
#include <SushiAI/vision/datasets/cifar.hpp>
#include <SushiAI/vision/datasets/classification_collator.hpp>
#include <SushiAI/vision/datasets/coco_json.hpp>
#include <SushiAI/vision/datasets/collator.hpp>
#include <SushiAI/vision/datasets/csv.hpp>
#include <SushiAI/vision/datasets/detection_collator.hpp>
#include <SushiAI/vision/datasets/fit_mask.hpp>
#include <SushiAI/vision/datasets/fit_sample.hpp>
#include <SushiAI/vision/datasets/fitted_sample.hpp>
#include <SushiAI/vision/datasets/idx.hpp>
#include <SushiAI/vision/datasets/idx_file.hpp>
#include <SushiAI/vision/datasets/image_extensions.hpp>
#include <SushiAI/vision/datasets/image_folder.hpp>
#include <SushiAI/vision/datasets/load_sample_image.hpp>
#include <SushiAI/vision/datasets/load_sample_mask.hpp>
#include <SushiAI/vision/datasets/mask_png.hpp>
#include <SushiAI/vision/datasets/mnist.hpp>
#include <SushiAI/vision/datasets/multi_label_collator.hpp>
#include <SushiAI/vision/datasets/reader_dataset.hpp>
#include <SushiAI/vision/datasets/reader_section.hpp>
#include <SushiAI/vision/datasets/reader_table.hpp>
#include <SushiAI/vision/datasets/regression_collator.hpp>
#include <SushiAI/vision/datasets/sample.hpp>
#include <SushiAI/vision/datasets/segmentation_collator.hpp>
#include <SushiAI/vision/datasets/stored_pixels.hpp>
#include <SushiAI/vision/datasets/table_dataset.hpp>
#include <SushiAI/vision/datasets/task_table.hpp>
#include <SushiAI/vision/datasets/value_table.hpp>
#include <SushiAI/vision/datasets/voc_xml.hpp>
#include <SushiAI/vision/datasets/xml_element.hpp>
#include <SushiAI/vision/datasets/yolo_txt.hpp>

#include <SushiAI/vision/codecs/decode_image.hpp>
#include <SushiAI/vision/codecs/decode_options.hpp>
#include <SushiAI/vision/codecs/image_format.hpp>
#include <SushiAI/vision/codecs/jpeg_decoder.hpp>
#include <SushiAI/vision/codecs/png_decoder.hpp>

#include <SushiAI/vision/registration/registration.hpp>

#include <SushiAI/vision/detect/anchor/anchor_clustering.hpp>
#include <SushiAI/vision/detect/anchor/anchor_count.hpp>
#include <SushiAI/vision/detect/anchor/anchor_fit.hpp>
#include <SushiAI/vision/detect/anchor/anchor_layout.hpp>
#include <SushiAI/vision/detect/anchor/anchor_review.hpp>
#include <SushiAI/vision/detect/anchor/anchor_size.hpp>
#include <SushiAI/vision/detect/anchor/anchor_task.hpp>
#include <SushiAI/vision/detect/anchor/head_training_section.hpp>
#include <SushiAI/vision/detect/anchor/assignment_thresholds.hpp>
#include <SushiAI/vision/detect/anchor/head_assignment.hpp>
#include <SushiAI/vision/detect/anchor/head_cost.hpp>
#include <SushiAI/vision/detect/anchor/head_decode.hpp>
#include <SushiAI/vision/detect/anchor/head_delta.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>
#include <SushiAI/vision/detect/anchor/head_training_options.hpp>
#include <SushiAI/vision/detect/anchor/seeded_sequence.hpp>
#include <SushiAI/vision/detect/suppression_criterion.hpp>

#include <SushiAI/vision/importers/darknet_config.hpp>
#include <SushiAI/vision/importers/darknet_fold.hpp>
#include <SushiAI/vision/importers/darknet_import.hpp>
#include <SushiAI/vision/importers/darknet_model.hpp>
#include <SushiAI/vision/importers/darknet_recipe.hpp>
#include <SushiAI/vision/importers/darknet_store.hpp>
#include <SushiAI/vision/importers/darknet_tensors.hpp>
#include <SushiAI/vision/importers/darknet_training.hpp>
#include <SushiAI/vision/importers/darknet_weights.hpp>

#include <SushiAI/vision/training/anchor_objective.hpp>
#include <SushiAI/vision/training/anchor_training_step.hpp>
#include <SushiAI/vision/training/truth_from_roles.hpp>
#include <SushiAI/vision/training/anchor_objective_factory.hpp>
#include <SushiAI/vision/training/anchor_training_loop.hpp>
#include <SushiAI/vision/training/head_delta_loss.hpp>
#include <SushiAI/vision/detect/iou_gradient.hpp>
#include <SushiAI/vision/detect/batch_average_precision.hpp>
#include <SushiAI/vision/detect/float_order.hpp>
#include <SushiAI/vision/detect/anchor/anchor_plan.hpp>
#include <SushiAI/vision/detect/anchor/head_truth.hpp>
#include <SushiAI/vision/detect/anchor/assignment_section.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_assignment.hpp>
#include <SushiAI/vision/detect/anchor/shape_ratio_delta.hpp>
#include <SushiAI/vision/detect/anchor/target_assignment.hpp>

#include <SushiAI/vision/blocks/segment_head.hpp>
#include <SushiAI/vision/blocks/c2f.hpp>
#include <SushiAI/vision/blocks/c3.hpp>
#include <SushiAI/vision/segment/argmax_classes.hpp>
#include <SushiAI/vision/segment/confusion_matrix.hpp>
#include <SushiAI/vision/segment/fill_outline.hpp>
#include <SushiAI/vision/segment/ignore_index.hpp>
#include <SushiAI/vision/segment/intersection_over_union.hpp>
#include <SushiAI/vision/segment/pixel_accuracy.hpp>
#include <SushiAI/vision/segment/rasterise_outlines.hpp>
#include <SushiAI/vision/training/dice_objective.hpp>
#include <SushiAI/vision/training/dice_targets.hpp>
#include <SushiAI/vision/training/pixel_cross_entropy_objective.hpp>
#include <SushiAI/vision/training/pixel_extents.hpp>
#include <SushiAI/vision/training/pixel_labels.hpp>
#include <SushiAI/vision/training/pixel_losses.hpp>
#include <SushiAI/vision/training/pixel_targets.hpp>
#include <SushiAI/vision/training/segmentation_metrics.hpp>
#include <SushiAI/vision/training/anchor_bias_init.hpp>
#include <SushiAI/vision/training/anchor_head.hpp>
#include <SushiAI/vision/training/best_shape_head_delta_rule.hpp>
#include <SushiAI/vision/training/detection_run_check.hpp>
#include <SushiAI/vision/training/detection_truth_check.hpp>
#include <SushiAI/vision/training/distribution_expectation_graph.hpp>
#include <SushiAI/vision/training/distribution_focal_loss.hpp>
#include <SushiAI/vision/training/free_assignment_stage.hpp>
#include <SushiAI/vision/training/free_batch_decoder.hpp>
#include <SushiAI/vision/training/free_bias_init.hpp>
#include <SushiAI/vision/training/free_detection_loss.hpp>
#include <SushiAI/vision/training/free_detection_loss_host.hpp>
#include <SushiAI/vision/training/free_loss_gains.hpp>
#include <SushiAI/vision/training/free_model_facts.hpp>
#include <SushiAI/vision/training/free_objective.hpp>
#include <SushiAI/vision/training/free_objective_factory.hpp>
#include <SushiAI/vision/training/free_training_loop.hpp>
#include <SushiAI/vision/training/free_training_step.hpp>
#include <SushiAI/vision/training/head_delta_rule.hpp>
#include <SushiAI/vision/training/head_delta_rule_choice.hpp>
#include <SushiAI/vision/training/run_progress.hpp>
#include <SushiAI/vision/training/shape_ratio_head_delta_rule.hpp>

#include <SushiAI/vision/datasets/augment_pipeline.hpp>
#include <SushiAI/vision/datasets/fit_pipeline.hpp>
#include <SushiAI/vision/datasets/pipeline_table.hpp>
#include <SushiAI/vision/datasets/polygon_mask.hpp>
#include <SushiAI/vision/datasets/sample_cache.hpp>
#include <SushiAI/vision/datasets/sample_pipeline.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/mask_pipeline.hpp>
