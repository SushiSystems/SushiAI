/**************************************************************************/
/* caffe_v1_types.cpp                                                     */
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
 * @file caffe_v1_types.cpp
 * @brief Defines the table that names the layer types of Caffe's V1 layer message.
 * @author Mustafa Garip
 */

#include "caffe_v1_types.hpp"

#include <cstdint>
#include <string_view>

namespace SushiAI
{
    namespace Interchange
    {
        namespace Detail
        {
            namespace
            {
                /** @brief Lists V1LayerParameter.LayerType with the names Caffe's upgrade gives. */
                const CaffeV1Type K_V1_TYPES[] = {
                    {1, "ACCURACY", "Accuracy"},
                    {2, "BNLL", "BNLL"},
                    {3, "CONCAT", "Concat"},
                    {4, "CONVOLUTION", "Convolution"},
                    {5, "DATA", "Data"},
                    {6, "DROPOUT", "Dropout"},
                    {7, "EUCLIDEAN_LOSS", "EuclideanLoss"},
                    {8, "FLATTEN", "Flatten"},
                    {9, "HDF5_DATA", "HDF5Data"},
                    {10, "HDF5_OUTPUT", "HDF5Output"},
                    {11, "IM2COL", "Im2col"},
                    {12, "IMAGE_DATA", "ImageData"},
                    {13, "INFOGAIN_LOSS", "InfogainLoss"},
                    {14, "INNER_PRODUCT", "InnerProduct"},
                    {15, "LRN", "LRN"},
                    {16, "MULTINOMIAL_LOGISTIC_LOSS", "MultinomialLogisticLoss"},
                    {17, "POOLING", "Pooling"},
                    {18, "RELU", "ReLU"},
                    {19, "SIGMOID", "Sigmoid"},
                    {20, "SOFTMAX", "Softmax"},
                    {21, "SOFTMAX_LOSS", "SoftmaxWithLoss"},
                    {22, "SPLIT", "Split"},
                    {23, "TANH", "TanH"},
                    {24, "WINDOW_DATA", "WindowData"},
                    {25, "ELTWISE", "Eltwise"},
                    {26, "POWER", "Power"},
                    {27, "SIGMOID_CROSS_ENTROPY_LOSS", "SigmoidCrossEntropyLoss"},
                    {28, "HINGE_LOSS", "HingeLoss"},
                    {29, "MEMORY_DATA", "MemoryData"},
                    {30, "ARGMAX", "ArgMax"},
                    {31, "THRESHOLD", "Threshold"},
                    {32, "DUMMY_DATA", "DummyData"},
                    {33, "SLICE", "Slice"},
                    {34, "MVN", "MVN"},
                    {35, "ABSVAL", "AbsVal"},
                    {36, "SILENCE", "Silence"},
                    {37, "CONTRASTIVE_LOSS", "ContrastiveLoss"},
                    {38, "EXP", "Exp"},
                    {39, "DECONVOLUTION", "Deconvolution"},
                };
            } // namespace

            const CaffeV1Type* find_v1_type_by_number(std::int64_t number) noexcept
            {
                for (const CaffeV1Type& row : K_V1_TYPES)
                    if (row.number == number)
                        return &row;
                return nullptr;
            }

            const CaffeV1Type* find_v1_type_by_identifier(std::string_view identifier) noexcept
            {
                for (const CaffeV1Type& row : K_V1_TYPES)
                    if (row.identifier == identifier)
                        return &row;
                return nullptr;
            }
        } // namespace Detail
    } // namespace Interchange
} // namespace SushiAI
