/**************************************************************************/
/*  main.cpp                                                              */
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

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <memory>
#include <numeric>
#include <utility>

#include <SushiAI/tensor>
#include <SushiAI/neuralnetwork>
#include <SushiAI/loss>
#include <SushiAI/optimizer>

using namespace SushiAI;

#pragma region -Disabled- Dataset (Surrogate Thin Airfoil Theory)

/*std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> dataset;
std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<float> distAoA(-5.0f, 15.0f);
std::uniform_real_distribution<float> distThickness(0.05f, 0.15f);
std::uniform_real_distribution<float> distCamber(0.0f, 0.1f);

for (int i = 0; i < 1000; ++i)
{
    float x1 = distAoA(gen);
    float x2 = distThickness(gen);
    float x3 = distCamber(gen);

    float y =
        0.1f * x1 * x1 +
        0.5f * x2 * x2 +
        2.0f * x3 * x3 +
        0.3f * x1 * x2 +
        0.2f * x1 * x3 +
        1.2f * x2 * x3 +
        0.4f * x1 +
        2.0f * x2 +
        5.0f * x3 +
        0.1f;

    auto input = std::make_shared<Tensor>(std::vector<int>{1, 3}, 0.0f, true);
    auto target = std::make_shared<Tensor>(std::vector<int>{1, 1}, y, false);

    input -> getData()[0] = x1;
    input -> getData()[1] = x2;
    input -> getData()[2] = x3;

    dataset.emplace_back(input, target);
}*/

#pragma endregion

#pragma region -Disabled- Dataset (Non-linear Quadratic Function)

/*std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> dataset;
std::mt19937 gen(std::random_device{}());
std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

for (int i = 0; i < 1000; ++i)
{
    float x1 = dist(gen);
    float x2 = dist(gen);
    float y = (x1 * x1) + x2;

    auto input = std::make_shared<Tensor>(std::vector<int>{1, 2}, 0.0f, true);
    auto target = std::make_shared<Tensor>(std::vector<int>{1, 1}, y, false);

    input -> getData()[0] = x1;
    input -> getData()[1] = x2;

    dataset.emplace_back(input, target);
}*/

#pragma endregion

int main()
{
    #pragma region Model (Neural Network Structure) : TSAGI 12

    auto model = std::make_shared<Sequential>();
    model -> add(std::make_shared<Linear>(1, 64, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model -> add(std::make_shared<LeakyReLU>(.01f));
    model -> add(std::make_shared<Linear>(64, 128, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model -> add(std::make_shared<Tanh>());
    model -> add(std::make_shared<Linear>(128, 64, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model -> add(std::make_shared<LeakyReLU>(.01f));
    model -> add(std::make_shared<Linear>(64, 1, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    #pragma endregion

    #pragma region Training The Model
    
    #pragma region -Disabled- Dataset (TSAGI-12 Surrogate)

    // Source: http://airfoiltools.com/airfoil/details?airfoil=tsagi12-il

    std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> dataset;
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> distAoA(-5.0f, 12.0f);

    for (int i = 0; i < 4000; ++i)
    {
        float x1 = distAoA(gen);

        //Surrogate function for lift coefficient (Cl) of TSAGI-12 
        float y =
            -0.000294117647058824f * x1 * x1 * x1
            + 0.00147058823529412f * x1 * x1
            + 0.124705882352941f * x1
            + 0.1f;

        auto input = std::make_shared<Tensor>(std::vector<int>{1, 1}, 0.0f, true);
        auto target = std::make_shared<Tensor>(std::vector<int>{1, 1}, y, false);

        input -> getData()[0] = x1;

        dataset.emplace_back(input, target);
    }

    #pragma endregion

    #pragma region Loss & Optimizer

    auto lossFunction = std::make_shared<MSELoss>();
    auto optimizer = std::make_shared<Adam>
    (
        0.000001f, //lr: learning rate
        0.9f,      //beta1: momentum of first moment
        0.999f,    //beta2: momentum of second moment (squared gradients)
        1e-8f      //epsilon: numerical stability
    );

    #pragma endregion

    #pragma region Training Loop

    int numberOfEpochs = 10;
    for (int epoch = 0; epoch < numberOfEpochs; ++epoch)
    {
        float totalLoss = 0.0f;

        #pragma region Dataset

        int i = 0;
        for (auto& [input, target] : dataset)
        {

            auto prediction = model -> forward(input);
            auto loss = lossFunction -> forward(prediction, target);
            totalLoss += loss -> getData()[0];

            loss -> backward();
            optimizer -> step(model -> parameters());

            if (i == 0 && epoch % 5 == 0)
                model -> printSummary();

            optimizer -> zeroGradient(model -> parameters());
            ++i;
        }

        #pragma endregion
        
        #pragma region Optimizer

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

        #pragma endregion
        
        float avgLoss = totalLoss / dataset.size();

        #pragma region Printing Epoch Summary

        //if (epoch % 10 == 9), YOU CAN USE IF NECESSARY.
        
        std::cout << "Epoch " << epoch + 1 << "/" << numberOfEpochs << " | Average Loss: " << std::fixed << std::setprecision(6) << avgLoss << " | Optimizer: " << optName << " | LR: " << lr;

        if (optName == "SGD")
            std::cout << " | Momentum: " << momentum;

        std::cout << "\n";

        int layerIndex = 0;
        for (auto& p : model -> parameters())
        {
            const auto& W = p -> getData();
            float sumW = 0;

            for (auto v : W)
                sumW += std::fabs(v);

            std::cout << "Layer [" << layerIndex++ << "] | |W| = " << sumW << "\n";
        }

        std::cout << std::endl;

        #pragma endregion
    }

    #pragma endregion

    #pragma region Save Model

    model -> saveModel("tsagi12.sushi");

    #pragma endregion

    #pragma endregion

    #pragma region Inferencing

    #pragma region -Disabled- Load Model

    //model -> loadModel("tsagi12.sushi");

    #pragma endregion

    #pragma region Inference

    std::cout << "\ ====== Inference ====== \n";

    auto aoaInput = std::make_shared<Tensor>(std::vector<int>{1, 1}, 0.0f, false);
    aoaInput -> getData()[0] = 7.5f;   

    auto prediction = model -> forward(aoaInput, false);

    prediction -> print("Lift Coefficient");

    #pragma endregion

    #pragma endregion

    return 0;
}
