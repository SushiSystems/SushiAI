#pragma once
#include "tensor.h"

namespace SushiAI 
{
	#pragma region Tensor Operations
	
	/// Add two tensors with each other, 
	std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);


	std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
	/// Matrix Multiplication
	std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
	std::shared_ptr<Tensor> slice(const std::shared_ptr<Tensor>& t, int index);

	#pragma endregion

	#pragma region Activation Functions

	/// ReLU Funtion, returns max(0, x).
	std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t);
	/// LeakyReLU Function, returns max(-alpha * x, x)
	std::shared_ptr<Tensor> leakyRelu(const std::shared_ptr<Tensor>& t, float alpha = 0.01f);
	/// Sigmoid function.
	std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t);
	/// Literally Tanh hyperbolic function.
	std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t);

	#pragma endregion

	#pragma region Loss Functions

	std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t);
	int argmax(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> crossEntropyLoss(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& targets);

	#pragma endregion
}
