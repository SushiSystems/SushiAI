#pragma once
#include "tensor.h"

namespace SushiAI 
{
	std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
	std::shared_ptr<Tensor> add_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> relu_cuda(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> sigmoid_cuda(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> tanh_cuda(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> softmax_cuda(const std::shared_ptr<Tensor>& t);
	std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
	std::shared_ptr<Tensor> matmul_cuda(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

	int argmax(const std::shared_ptr<Tensor>& t);
	float cross_entropy_loss(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& targets);;
}
