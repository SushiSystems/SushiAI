#include "tensor.h"
#include "ops.h"
// Eðer ops.cu ayrý derleniyorsa burada ayrý bir header eklemen gerekebilir
// #include "ops_cuda.h"
#include <iostream>
#include <memory>

int main() {
    using namespace SushiAI;
    std::cout << "===== SushiAI Autograd/Pointer Test =====\n";

    // CPU - Add
    auto a = std::make_shared<Tensor>(std::vector<int>{2, 2}, 1.0f, true);
    auto b = std::make_shared<Tensor>(std::vector<int>{2, 2}, 2.0f, true);
    auto c = add(a, b);
    c->print("CPU Add (a+b)");
    c->backward();
    a->print("a (grad)");
    b->print("b (grad)");

    // CPU - ReLU
    auto relu_in = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    relu_in->data = { -1.0f, 0.0f, 2.0f, 4.0f };
    auto relu_out = relu(relu_in);
    relu_out->print("CPU ReLU");
    relu_out->backward();
    relu_in->print("ReLU Input (grad)");

    // CPU - Sigmoid
    auto sig_in = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    sig_in->data = { -1.0f, 0.0f, 1.0f, 2.0f };
    auto sig_out = sigmoid(sig_in);
    sig_out->print("CPU Sigmoid");
    sig_out->backward();
    sig_in->print("Sigmoid Input (grad)");

    // CPU - Tanh
    auto tanh_in = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    tanh_in->data = { -1.0f, 0.0f, 1.0f, 2.0f };
    auto tanh_out = tanh(tanh_in);
    tanh_out->print("CPU Tanh");
    tanh_out->backward();
    tanh_in->print("Tanh Input (grad)");

    // CPU - Softmax
    auto sm_in = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    sm_in->data = { 1.0f, 2.0f, 3.0f, 4.0f };
    auto sm_out = softmax(sm_in);
    sm_out->print("CPU Softmax");
    sm_out->backward();
    sm_in->print("Softmax Input (grad)");

    // CPU - Matmul
    auto m1 = std::make_shared<Tensor>(std::vector<int>{2, 3}, 0.0f, true);
    auto m2 = std::make_shared<Tensor>(std::vector<int>{3, 2}, 0.0f, true);
    m1->data = { 1,2,3,4,5,6 };
    m2->data = { 7,8,9,10,11,12 };
    auto mat_out = matmul(m1, m2);
    mat_out->print("CPU Matmul");
    mat_out->backward();
    m1->print("Matmul m1 (grad)");
    m2->print("Matmul m2 (grad)");

    std::cout << "\n===== CUDA Testler =====\n";

    // CUDA - Add
    auto a_cuda = std::make_shared<Tensor>(std::vector<int>{2, 2}, 1.0f, true);
    auto b_cuda = std::make_shared<Tensor>(std::vector<int>{2, 2}, 2.0f, true);
    auto c_cuda = add_cuda(a_cuda, b_cuda);
    c_cuda->print("CUDA Add (a+b)");
    c_cuda->backward();
    a_cuda->print("a_cuda (grad)");
    b_cuda->print("b_cuda (grad)");

    // CUDA - ReLU
    auto relu_in_cuda = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    relu_in_cuda->data = { -1.0f, 0.0f, 2.0f, 4.0f };
    auto relu_out_cuda = relu_cuda(relu_in_cuda);
    relu_out_cuda->print("CUDA ReLU");
    relu_out_cuda->backward();
    relu_in_cuda->print("ReLU Input CUDA (grad)");

    // CUDA - Sigmoid
    auto sig_in_cuda = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    sig_in_cuda->data = { -1.0f, 0.0f, 1.0f, 2.0f };
    auto sig_out_cuda = sigmoid_cuda(sig_in_cuda);
    sig_out_cuda->print("CUDA Sigmoid");
    sig_out_cuda->backward();
    sig_in_cuda->print("Sigmoid Input CUDA (grad)");

    // CUDA - Tanh
    auto tanh_in_cuda = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    tanh_in_cuda->data = { -1.0f, 0.0f, 1.0f, 2.0f };
    auto tanh_out_cuda = tanh_cuda(tanh_in_cuda);
    tanh_out_cuda->print("CUDA Tanh");
    tanh_out_cuda->backward();
    tanh_in_cuda->print("Tanh Input CUDA (grad)");

    // CUDA - Softmax
    auto sm_in_cuda = std::make_shared<Tensor>(std::vector<int>{4}, 0.0f, true);
    sm_in_cuda->data = { 1.0f, 2.0f, 3.0f, 4.0f };
    auto sm_out_cuda = softmax_cuda(sm_in_cuda);
    sm_out_cuda->print("CUDA Softmax");
    sm_out_cuda->backward();
    sm_in_cuda->print("Softmax Input CUDA (grad)");

    // CUDA - Matmul
    auto m1_cuda = std::make_shared<Tensor>(std::vector<int>{2, 3}, 0.0f, true);
    auto m2_cuda = std::make_shared<Tensor>(std::vector<int>{3, 2}, 0.0f, true);
    m1_cuda->data = { 1,2,3,4,5,6 };
    m2_cuda->data = { 7,8,9,10,11,12 };
    auto mat_out_cuda = matmul_cuda(m1_cuda, m2_cuda);
    mat_out_cuda->print("CUDA Matmul");
    mat_out_cuda->backward();
    m1_cuda->print("Matmul m1 CUDA (grad)");
    m2_cuda->print("Matmul m2 CUDA (grad)");

    std::cout << "\nTüm testler baþarýyla tamamlandý!\n";
    return 0;
}
