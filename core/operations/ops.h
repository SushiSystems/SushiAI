/**************************************************************************/
/*  ops.h                                                                 */
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

#include "tensor.h"

namespace SushiAI 
{
	#pragma region Tensor Operations
	
	/// Element-wise addition between tensors a and b, with broadcasting support.
	std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	/// Element-wise multiplication between tensors a and b, with broadcasting support.
	std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	/// Computes the matrix product of two matrices (2D tensors) a and b.
	std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	/// Extracts a single slice from the first (batch) dimension of a 3D tensor.
	std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& t, int index);

	#pragma endregion

	#pragma region Activation Functions

	/// Applies the rectified linear unit: f(x) = max(0, x).
	std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t);

	/// Applies f(x) = max(alpha * x, x) to allow small gradients when x < 0.
	std::shared_ptr<Tensor> leakyRelu(const std::shared_ptr<Tensor>& t, float alpha = 0.01f);

	/// Applies f(x) = 1 / (1 + exp(-x)) to squash input into the (0, 1) range.
	std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t);

	/// Applies the hyperbolic tangent function: f(x) = tanh(x), output intervals (-1, 1).
	std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t);

	#pragma endregion

	#pragma region Loss Functions

	/// Applies the softmax operation along the last dimension to convert logits into probabilities.
	std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t);
	
	/// Returns the index of the maximum value along the last dimension of the tensor.
	int argmax(const std::shared_ptr<Tensor>& t);

	/// Computes the categorical cross-entropy loss between logits and targets.
	std::shared_ptr<Tensor> crossEntropyLoss(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& targets);

	#pragma endregion
}
