#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <memory>
#include <numeric>
#include <utility>
#include "initializer.h"
#include "sequential.h"
#include "optimizer.h"
#include "tensor.h"
#include "layer.h"
#include "loss.h"
#include "ops.h"

using namespace SushiAI;

/*int main()
{
    std::shared_ptr<PhysicsProblem> problem = std::make_shared<PoissonProblem>();

    // 2️⃣ Model Tanımı (örnek: 2 giriş, 1 çıkış)
    auto model = std::make_shared<Sequential>();

    // Giriş katmanı: 2 → 50
    model->add(std::make_shared<Linear>(2, 100, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model->add(std::make_shared<Tanh>());

    // 3 adet gizli katman: 50 → 50
    for (int i = 0; i < 5; ++i)
    {
        model->add(std::make_shared<Linear>(100, 100, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
        model->add(std::make_shared<Tanh>());
    }

    // Çıkış katmanı: 50 → 1 (skaler u)
    model->add(std::make_shared<Linear>(100, 1, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));

    // 3️⃣ Loss ve Optimizer
    auto loss = std::make_shared<MSELoss>();
    auto optimizer = std::make_shared<Adam>(
        .001f,   // learning rate (lr)
        0.9f,     // beta1: momentum of first moment
        0.999f,   // beta2: momentum of second moment (squared gradients)
        1e-8f     // epsilon: numerical stability
    );

    // 4️⃣ PhysicsSolver ile eğitim başlat
    PhysicsSolver solver(problem, model, loss, optimizer, 200, 200, 500); // interior=1000, boundary=200, epochs=500

    solver.train();
}*/

int main()
{
    std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> dataset;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int i = 0; i < 1000; ++i) 
    {
        float x1 = dist(gen); // [-1, 1]
        float x2 = dist(gen);
        float y = (x1 * x1) + x2;

        auto input = std::make_shared<Tensor>(std::vector<int>{1, 2}, 0.0f, true);
        auto target = std::make_shared<Tensor>(std::vector<int>{1, 1}, y, false);

        input->getData()[0] = x1;
        input->getData()[1] = x2;

        dataset.emplace_back(input, target);
    }

    // === 2) Model
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(2, 16, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model->add(std::make_shared<Linear>(16, 16, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model->add(std::make_shared<Tanh>());
    model->add(std::make_shared<Linear>(16, 1, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));

    // === 3) Loss & Optimizer
    auto lossFunction = std::make_shared<MSELoss>();
    auto optimizer = std::make_shared<Adam>(
        0.001f,   // learning rate (lr)
        0.9f,     // beta1: momentum of first moment
        0.999f,   // beta2: momentum of second moment (squared gradients)
        1e-8f     // epsilon: numerical stability
    );

    // === 4) Training Loop
    int numEpochs = 10;
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        float totalLoss = 0.0f;

        // 1) Tüm dataset üzerinde bir tur
        for (auto& [input, target] : dataset)
        {
            auto prediction = model->forward(input);
            auto loss = lossFunction->forward(prediction, target);
            totalLoss += loss->getData()[0];

            // Backward + step
            loss->backward();
            std::cout << "dW0: " << model->parameters()[0]->getGradient()[0] << "\n";
            optimizer->step(model->parameters());
            optimizer->zeroGradient(model->parameters());
        }

        // 2) Öğrenme hızı / optimizer bilgisi
        float lr = 0.0f;
        float momentum = 0.0f;
        std::string optName = "Unknown";
        if (auto sgd = std::dynamic_pointer_cast<SGD>(optimizer))
        {
            lr = sgd -> getLearningRate();
            momentum = sgd -> getMomentum();
            optName = "SGD";
        }
        else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer))
        {
            lr = adam -> getLearningRate();
            optName = "Adam";
        }

        // 3) Epoch sonucu: loss + optimizer
        float avgLoss = totalLoss / dataset.size();

        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs
            << " | Avg Loss: " << std::fixed << std::setprecision(6) << avgLoss
            << " | Optimizer: " << optName
            << " | LR: " << lr;
        if (optName == "SGD") std::cout << " | Momentum: " << momentum;

        std::cout << "\n";

        int layer_idx = 0;
        for (auto& p : model -> parameters()) 
        {
            const auto& W = p -> getData();
            const auto& dW = p -> getGradient();
            float sumW = 0, sumGrad = 0;

            for (auto v : W)    
                sumW += std::fabs(v);

            for (auto g : dW)   
                sumGrad += std::fabs(g);

            std::cout << "Param[" << layer_idx++ << "] | |W|=" << sumW << " | |dW|=" << sumGrad << "\n";
        }

        std::cout << std::endl;
    }

    return 0;
}

/*
void print_header(const char* msg)
{
    std::cout << "\n===== " << msg << " =====\n";
}

int main()
{
    // 1) Element-wise 1D add
    print_header("1D Elementwise Add");
    auto a1 = std::make_shared<Tensor>(std::vector<int>{3}, 0.0f, true);
    a1->getData() = { 1, 2, 3 };
    auto b1 = std::make_shared<Tensor>(std::vector<int>{3}, 0.0f, true);
    b1->getData() = { 4, 5, 6 };
    auto c1 = add(a1, b1);
    c1->print("c1 (forward)");
    c1->backward();
    a1->print("a1.grad");
    b1->print("b1.grad");

    // 2) 2D Element-wise add
    print_header("2D Elementwise Add");
    auto a2 = std::make_shared<Tensor>(std::vector<int>{2, 2}, 0.0f, true);
    auto b2 = std::make_shared<Tensor>(std::vector<int>{2, 2}, 0.0f, true);
    a2->getData() = { 1,2,3,4 };
    b2->getData() = { 10,20,30,40 };
    auto c2 = add(a2, b2);
    c2->print("c2");
    c2->backward();
    a2->print("a2.grad");
    b2->print("b2.grad");

    // 3) 2D + 1D row-broadcast
    print_header("2D + 1D Row-Broadcast Add");
    auto a3 = std::make_shared<Tensor>(std::vector<int>{3, 2}, 0.0f, true);
    a3->getData() = { 1, 2,  3, 4,  5, 6 };
    auto b3 = std::make_shared<Tensor>(std::vector<int>{2}, 0.0f, true);
    b3->getData() = { 100, 200 };
    auto c3 = add(a3, b3);
    c3->print("c3");
    c3->backward();
    a3->print("a3.grad");
    b3->print("b3.grad");

    // 4) 1D + 2D col-broadcast Add
    print_header("1D + 2D Col-Broadcast Add");
    auto a4 = std::make_shared<Tensor>(std::vector<int>{2}, 0.0f, true);
    a4->getData() = { 7, 8 };
    auto b4 = std::make_shared<Tensor>(std::vector<int>{3, 2}, 0.0f, true);
    b4->getData() = { 10,20, 30,40, 50,60 };
    auto c4 = add(a4, b4);
    c4->print("c4");
    c4->backward();
    a4->print("a4.grad");
    b4->print("b4.grad");

    // 5) 3D + 1D broadcast Add
    print_header("3D + 1D Broadcast Add");
    auto a5 = std::make_shared<Tensor>(std::vector<int>{2, 2, 2}, 0.0f, true);
    // veri = [[[1,2],[3,4]], [[5,6],[7,8]]]
    a5->getData() = { 1,2,3,4, 5,6,7,8 };
    auto b5 = std::make_shared<Tensor>(std::vector<int>{2}, 0.0f, true);
    b5->getData() = { 1000, 2000 };
    auto c5 = add(a5, b5);
    c5->print("c5");
    c5->backward();
    a5->print("a5.grad");
    b5->print("b5.grad");

    // --- MatMul tests ---

    // 6) 2D MatMul
    print_header("2D MatMul");
    auto m1 = std::make_shared<Tensor>(std::vector<int>{2, 3}, 0.0f, true);
    auto m2 = std::make_shared<Tensor>(std::vector<int>{3, 2}, 0.0f, true);
    m1->getData() = { 1,2,3,4,5,6 };
    m2->getData() = { 7,8,9,10,11,12 };
    auto m3 = matmul(m1, m2);
    m3->print("m3");
    m3->backward();
    m1->print("m1.grad");
    m2->print("m2.grad");

    // 7) 3D Batch MatMul
    print_header("3D Batch MatMul");
    auto A = std::make_shared<Tensor>(std::vector<int>{2, 2, 2}, 0.0f, true);
    auto B = std::make_shared<Tensor>(std::vector<int>{2, 2, 2}, 0.0f, true);
    // batch 0: [[1,2],[3,4]]; batch1: [[5,6],[7,8]]
    A->getData() = { 1,2,3,4, 5,6,7,8 };
    B->getData() = { 2,0, 1,3, 4,1, 0,2 };
    auto Cbatch = matmul(A, B);
    Cbatch->print("Cbatch");
    Cbatch->backward();
    A->print("A.grad");
    B->print("B.grad");

    return 0;
}*/

/*
int main()
{
    using namespace SushiAI;

    std::cout << "===== SushiAI Autograd/Pointer Test =====\n";

    // CPU - Add
    auto a = std::make_shared<Tensor>(std::vector<int>{2, 2}, 1.0f, true);
    auto b = std::make_shared<Tensor>(std::vector<int>{2}, 2.0f, true);
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

    std::cout << "\nTestler başarıyla tamamlandı!\n";
    return 0;
}*/
