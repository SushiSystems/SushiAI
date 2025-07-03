/**************************************************************************/
/*  loss.h                                                                */
/**************************************************************************/
/*                          This file is part of:                         */
/*                                 SushiAI                                */
/*                 https://github.com/SushiSystems/SushiAI                */
/*                         https://sushisystems.io                        */
/**************************************************************************/
/* Copyright (c) 2025-present  Mustafa Garip & Sushi Systems              */
/*                                                                   	  */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include <memory>

#include "tensor.h"

namespace SushiAI 
{
    #pragma region Loss Class

    /// Abstract base class for loss functions.
    class Loss 
    {
        public:
            virtual ~Loss() = default;
            virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) = 0;
    };

    #pragma endregion

    #pragma region Loss Functions

    /// Computes the average of squared differences between predictions and targets. (1 / n) * sum ((y_predicted - y_true) ^ 2)
    class MSELoss : public Loss 
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) override;
    };

    /// Applies softmax and computes the negative log likelihood between predicted logits and targets.
    class CrossEntropyLoss : public Loss 
    {
        public:
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, const std::shared_ptr<Tensor>& target) override;
    };

    #pragma endregion
}