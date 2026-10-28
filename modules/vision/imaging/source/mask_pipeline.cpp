/**************************************************************************/
/* mask_pipeline.cpp                                                      */
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
 * @file mask_pipeline.cpp
 * @brief Implements the pipeline that gives a mask the geometry of its training sample.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/imaging/mask_pipeline.hpp>

#include <string_view>


#include <SushiAI/seam/seam.hpp>
#include <SushiAI/vision/imaging/affine.hpp>
#include <SushiAI/vision/imaging/affine_nearest.hpp>
#include <SushiAI/vision/imaging/flip.hpp>
#include <SushiAI/vision/imaging/resize.hpp>

namespace SushiAI::Vision::Imaging
{
    namespace
    {
        /** @brief Names this unit in a message. */
        constexpr std::string_view K_PIPELINE = "TrainingMaskPipeline";

        /** @brief Holds the channel count of a mask. */
        constexpr std::int32_t K_MASK_CHANNELS = 1;
    } // namespace

    TrainingMaskPipeline::TrainingMaskPipeline(const MaskSource& source,
                                               const TrainingSettings& settings,
                                               std::uint8_t void_value) noexcept
        : source_(&source), settings_(settings), void_value_(void_value)
    {

    }

    Status TrainingMaskPipeline::load_resized(std::size_t index, Image& loaded)
    {
        SA_TRY(source_->load(index, decoded_));
        SA_FAIL_IF(!is_valid_size(decoded_.size()) || decoded_.channels() != K_MASK_CHANNELS,
                   StatusCode::Incompatible,
                   "{}: the mask of image {} is {}x{} of {} channels; a mask holds one class "
                   "index per pixel",
                   K_PIPELINE, index, decoded_.height(), decoded_.width(), decoded_.channels());
        SA_FAIL_IF(!is_valid_side(settings_.image_size), StatusCode::InvalidDescription,
                   "{}: an image_size of {} is outside [1, {}]", K_PIPELINE,
                   settings_.image_size, K_MAX_IMAGE_SIDE);

        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            resize_nearest(decoded_.view(),
                           fit_long_side(decoded_.size(), settings_.image_size), loaded);
            return Status();
        });
    }

    Status TrainingMaskPipeline::make_mask(std::size_t index, const TrainingTrace& trace,
                                           ImageSize loaded_size, Image& mask)
    {
        SA_TRY(load_resized(index, patches_[0]));
        SA_FAIL_IF(patches_[0].size() != loaded_size, StatusCode::Incompatible,
                   "{}: the mask of image {} is loaded at {}x{}; its image at {}x{}", K_PIPELINE,
                   index, patches_[0].height(), patches_[0].width(), loaded_size.height,
                   loaded_size.width);

        ImageView warp_source = patches_[0].view();
        if (trace.mosaic.is_enabled)
        {
            for (std::size_t i = 1; i < K_MOSAIC_PATCHES; ++i)
                SA_TRY(load_resized(trace.mosaic.partners[i - 1], patches_[i]));

            SA_TRY(SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
            {
                std::array<ImageSize, K_MOSAIC_PATCHES> sizes;
                std::array<ImageView, K_MOSAIC_PATCHES> views;
                for (std::size_t i = 0; i < K_MOSAIC_PATCHES; ++i)
                {
                    sizes[i] = patches_[i].size();
                    views[i] = patches_[i].view();
                }

                SA_TRY_ASSIGN(MosaicPlan plan, plan_mosaic(settings_.mosaic, trace.mosaic,
                                                           settings_.image_size, sizes));
                plan.fill_value = void_value_;
                return apply_mosaic(views, plan, canvas_);
            }));
            warp_source = canvas_.view();
        }

        AffinePlan warp = trace.affine;
        warp.border_value = void_value_;
        SA_TRY(apply_affine_nearest(warp_source, warp, warped_));

        SA_FAIL_IF(warped_.size() != trace.flip.source, StatusCode::Incompatible,
                   "{}: the flip was planned for {}x{}; the warped mask is {}x{}", K_PIPELINE,
                   trace.flip.source.height, trace.flip.source.width, warped_.height(),
                   warped_.width());

        return SushiAI::Detail::guard(SushiAI::Detail::SeamPhase::Allocate, [&]()
        {
            apply_flip(warped_.view(), trace.flip, mask);
            return Status();
        });
    }
} // namespace SushiAI::Vision::Imaging
