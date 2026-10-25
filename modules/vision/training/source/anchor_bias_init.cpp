/**************************************************************************/
/* anchor_bias_init.cpp                                                   */
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
 * @file anchor_bias_init.cpp
 * @brief Implements the stride-derived start of an anchor head's objectness and class biases.
 * @author Mustafa Garip
 */

#include <SushiAI/vision/training/anchor_bias_init.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

#include <SushiBLAS/core/common.hpp>
#include <SushiBLAS/tensor/tensor.hpp>
#include <SushiAI/shape/tensor_shape.hpp>
#include <SushiAI/vision/detect/anchor/head_options.hpp>

namespace SushiAI::Vision::Training
{
    namespace
    {
        /** @brief Holds the numerator of the objectness bias, the objects per image assumed. */
        constexpr double K_OBJECTS_PER_IMAGE = 8.0;

        /** @brief Holds the image side, in pixels, the objectness bias assumes. */
        constexpr double K_REFERENCE_IMAGE_SIDE = 640.0;

        /** @brief Holds the numerator of the class bias. */
        constexpr double K_CLASS_NUMERATOR = 0.6;

        /** @brief Holds what is taken off the class count in the class bias. */
        constexpr double K_CLASS_OFFSET = 0.99;

        /** @brief Holds the rank of a published head output, [B, H, W, channels]. */
        constexpr std::int32_t K_PORT_RANK = 4;

        /** @brief Holds the rank of one input sample, [H, W, C]. */
        constexpr std::int32_t K_SAMPLE_RANK = 3;

        /** @brief Names the row type whose activation a head may end in. */
        constexpr std::string_view K_ACTIVATION_ROW = "sigmoid";

        /** @brief Names the row type that holds the head's biased convolution. */
        constexpr std::string_view K_CONVOLUTION_ROW = "conv2d";

        /** @brief Holds one validated write: a bias tensor and the slot geometry to add to. */
        struct BiasWrite
        {
            SushiBLAS::Tensor* tensor; /**< The bias, float32 of slots * (5 + classes). */
            std::int64_t slots;        /**< Anchor slots in the tensor. */
            float objectness;          /**< Added to the objectness entry of every slot. */
            float classes;             /**< Added to every class entry of every slot. */
        };

        /** @brief Returns the published name an output entry answers to. */
        const std::string& published_name(const Config::OutputSpec& output) noexcept
        {
            return output.name.empty() ? output.port : output.name;
        }

        /** @brief Returns the row whose output is named @p name, or -1 when none is. */
        std::int64_t output_row(const Config::ModelSpec& spec, const std::string& name) noexcept
        {
            for (const Config::OutputSpec& output : spec.outputs)
            {
                if (published_name(output) == name)
                    return output.node;
            }
            return -1;
        }

        /**
         * @brief Returns the row of the biased convolution that head output @p name ends in.
         * @return InvalidDescription when the output is not published, ends in a row that is
         *         neither a sigmoid over a conv2d nor a conv2d, or the conv2d has no bias.
         */
        Result<std::int64_t> convolution_row(const Config::ModelSpec& spec,
                                             const std::string& name)
        {
            std::int64_t row = output_row(spec, name);
            SA_FAIL_IF(row < 0 || row >= static_cast<std::int64_t>(spec.layers.size()),
                       StatusCode::InvalidDescription,
                       "the model file publishes no output '{}' that names a row", name);

            const Config::LayerSpec& last = spec.layers[static_cast<std::size_t>(row)];
            if (last.type == K_ACTIVATION_ROW)
            {
                SA_FAIL_IF(last.from.size() != 1 || last.from.front().node < 0,
                           StatusCode::InvalidDescription,
                           "row {} of output '{}' must read one earlier row", row, name);
                row = last.from.front().node;
            }

            const Config::LayerSpec& layer = spec.layers[static_cast<std::size_t>(row)];
            SA_FAIL_IF(layer.type != K_CONVOLUTION_ROW, StatusCode::InvalidDescription,
                       "the output '{}' ends in row {} of type '{}', and a head ends in a "
                       "'{}' row", name, row, layer.type, K_CONVOLUTION_ROW);
            SA_FAIL_IF(layer.attributes.has("use_bias") &&
                           !layer.attributes.at("use_bias", "use_bias").as_bool("use_bias"),
                       StatusCode::InvalidDescription,
                       "row {} has no bias, so output '{}' has none to start", row, name);
            return row;
        }

        /** @brief Returns the stride of @p shape over @p input, which must divide it. */
        Result<std::int64_t> level_stride(const std::string& name, const Shape& shape,
                                          const Shape& input)
        {
            const std::int64_t height = shape.dim(1);
            const std::int64_t width = shape.dim(2);
            SA_FAIL_IF(height < 1 || width < 1 || input.dim(0) % height != 0
                           || input.dim(1) % width != 0
                           || input.dim(0) / height != input.dim(1) / width,
                       StatusCode::InvalidDescription,
                       "the output '{}' is {} by {} over an input of {} by {}; a level's stride "
                       "must divide the input exactly and be equal along H and W",
                       name, height, width, input.dim(0), input.dim(1));
            return input.dim(0) / height;
        }

        /** @brief Checks the bias entry of @p head in @p state and records its write. */
        Status plan_write(NN::ModelState& state, const AnchorBiasHead& head, std::int64_t slots,
                          std::int64_t classes, std::vector<BiasWrite>& writes)
        {
            const Result<SushiBLAS::Tensor*> found = state.tensor(head.bias);
            SA_FAIL_IF(!found.has_value(), StatusCode::InvalidDescription,
                       "the model state has no entry '{}', which the head's last convolution "
                       "declares", head.bias);
            SushiBLAS::Tensor* const tensor = found.value();

            SA_FAIL_IF(tensor->dtype != SushiBLAS::Core::DataType::Float32,
                       StatusCode::InvalidDescription, "'{}' is not float32", head.bias);
            const std::int64_t channels = Detect::Anchor::K_FIRST_CLASS_ENTRY + classes;
            const Shape actual = shape_of(*tensor);
            SA_FAIL_IF(actual != Shape{slots * channels}, StatusCode::InvalidDescription,
                       "'{}' is {}, but {} slots of {} entries give [{}]", head.bias,
                       actual.to_string(), slots, channels, slots * channels);
            SA_FAIL_IF(tensor->data_or_null() == nullptr, StatusCode::InvalidDescription,
                       "'{}' has no storage to receive its start", head.bias);

            writes.push_back(BiasWrite{tensor, slots,
                                       static_cast<float>(anchor_objectness_bias(head.stride)),
                                       static_cast<float>(anchor_class_bias(classes))});
            return Status();
        }

        /** @brief Adds the biases of @p write to every slot of its tensor. */
        void apply(const BiasWrite& write, std::int64_t classes)
        {
            const std::int64_t channels = Detect::Anchor::K_FIRST_CLASS_ENTRY + classes;
            float* const data = write.tensor->data_as<float>();
            for (std::int64_t slot = 0; slot < write.slots; ++slot)
            {
                float* const entries = data + slot * channels;
                entries[Detect::Anchor::K_OBJECTNESS_ENTRY] += write.objectness;
                for (std::int64_t entry = Detect::Anchor::K_FIRST_CLASS_ENTRY; entry < channels;
                     ++entry)
                    entries[entry] += write.classes;
            }
        }
    } // namespace

    double anchor_objectness_bias(std::int64_t stride)
    {
        const double cells = K_REFERENCE_IMAGE_SIDE / static_cast<double>(stride);
        return std::log(K_OBJECTS_PER_IMAGE / (cells * cells));
    }

    double anchor_class_bias(std::int64_t classes)
    {
        return std::log(K_CLASS_NUMERATOR / (static_cast<double>(classes) - K_CLASS_OFFSET));
    }

    Result<std::vector<AnchorBiasHead>> derive_anchor_bias_heads(
        const Config::ModelSpec& spec, const Shape& input_sample,
        const Detect::Anchor::AnchorTask& task, const std::vector<Train::OutputShape>& outputs)
    {
        SA_FAIL_IF(input_sample.rank() != K_SAMPLE_RANK, StatusCode::InvalidDescription,
                   "the input sample is {}; it must be [H, W, C]", input_sample.to_string());

        std::vector<AnchorBiasHead> heads;
        heads.reserve(task.heads.size());
        for (const Detect::Anchor::AnchorTaskHead& head : task.heads)
        {
            const Train::OutputShape* const output =
                Train::find_output_shape(outputs, head.output);
            SA_FAIL_IF(output == nullptr, StatusCode::InvalidDescription,
                       "the model does not publish the output '{}' that its task names",
                       head.output);
            SA_FAIL_IF(output->shape.rank() != K_PORT_RANK, StatusCode::InvalidDescription,
                       "the output '{}' is {}; a head is [B, H, W, channels]", head.output,
                       output->shape.to_string());

            SA_TRY_ASSIGN(const std::int64_t row, convolution_row(spec, head.output));
            SA_TRY_ASSIGN(const std::int64_t stride,
                          level_stride(head.output, output->shape, input_sample));
            heads.push_back(AnchorBiasHead{std::to_string(row).append(".bias"), stride});
        }
        return heads;
    }

    Status initialize_anchor_biases(const Detect::Anchor::AnchorTask& task,
                                    const std::vector<AnchorBiasHead>& heads,
                                    NN::ModelState& state)
    {
        SA_FAIL_IF(heads.size() != task.layout.head_anchors.size(),
                   StatusCode::InvalidDescription,
                   "{} bias heads were given, and the task has {} heads", heads.size(),
                   task.layout.head_anchors.size());

        std::vector<BiasWrite> writes;
        for (std::size_t head = 0; head < heads.size(); ++head)
        {
            SA_FAIL_IF(heads[head].stride < 1, StatusCode::InvalidDescription,
                       "head {} has the stride {}; it must be at least 1", head,
                       heads[head].stride);
            SA_TRY(plan_write(state, heads[head],
                              static_cast<std::int64_t>(task.layout.head_anchors[head].size()),
                              task.num_classes, writes));
        }

        // Every entry was checked above, so the writes below change all of them or none.
        for (const BiasWrite& write : writes)
            apply(write, task.num_classes);
        return Status();
    }

} // namespace SushiAI::Vision::Training
