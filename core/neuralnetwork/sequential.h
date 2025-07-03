/**************************************************************************/
/*  sequential.h                                                          */
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
#include <vector>
#include <memory>
#include <string>
#include <iomanip> 
#include <numeric>
#include <iostream>

#include "layer.h"

namespace SushiAI 
{
    #pragma region Sequential Class

    /// Sequential container for stacking multiple layers. Inherits from Layer and allows layers to be chained in order.
    class Sequential : public Layer 
    {
        public:
            Sequential() = default;

            #pragma region Sequential Constructor

            /// Constructs a Sequential model composed of a list of layers. Layers be executed in the order they appear in the vector.
            Sequential(const std::vector<std::shared_ptr<Layer>>& layers);

            #pragma endregion

            #pragma region Sequential Functions

            /// Performs a forward pass through all layers in the sequence.
            std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input, bool training = true) override;
            /// Name of the layer structure.
            std::string name() const override { return "Sequential"; }
            /// Returns all trainable parameters from each layer in sequence.
            std::vector<std::shared_ptr<Tensor>> parameters() const override;

            /// Adds a new layer to the end of the sequence.
            void add(const std::shared_ptr<Layer>& layer);
            /// Removes the layer at the specified index in the sequence.
            void remove(size_t index);

            #pragma endregion

            #pragma region Save / Load and 

            void saveModel(const std::string& path) const;
            void loadModel(const std::string& path);

            #pragma endregion

            #pragma region Printing Summary

            void printSummary() const
            {
                std::cout << "================ Model Summary ================\n\n";

                size_t totalParams = 0;
                for (size_t i = 0; i < layers.size(); ++i)
                {
                    auto& layer = layers[i];
                    size_t layerParamCount = 0;

                    std::cout << "[" << i << "] " << layer -> name() << ":\n";

                    auto params = layer -> parameters();
                    for (size_t j = 0; j < params.size(); ++j)
                    {
                        auto& t = params[j];
                        size_t n = t -> getTotalSize();
                        layerParamCount += n;

                        std::cout << "   Parameter #" << j << " Shape: ";

                        for (int d : t -> getShape())
                            std::cout << "[" << d << "]";

                        std::cout << " | Count: " << n;

                        const auto& grad = t -> getGradient();
                        float gradSum = std::accumulate(grad.begin(), grad.end(), 0.0f, [](float acc, float g) { return acc + std::abs(g); });

                        std::cout << " | Gradient Sum: " << std::fixed << std::setprecision(6) << gradSum;
                        std::cout << " | Gradient[0..2]: ";

                        for (int k = 0; k < std::min(3, (int)grad.size()); ++k)
                            std::cout << std::fixed << std::setprecision(4) << grad[k] << (k < 2 ? ", " : "");
                        
                        std::cout << "\n";
                    }

                    totalParams += layerParamCount;

                    std::cout << " --> Layer total parameters: " << layerParamCount << "\n\n";
                }

                std::cout << "=== Total trainable parameters: " << totalParams << " ===\n\n";
                std::cout << "===================== * * =====================\n\n";
            }

            #pragma endregion

            #pragma region Getters

            std::shared_ptr<Layer> getLayer(size_t index) const
            {
                if (index < layers.size())
                    return layers[index];

                return nullptr;
            }
            size_t getLayersSize() const { return layers.size(); }

            #pragma endregion

        private:
            std::vector<std::shared_ptr<Layer>> layers;
    };

    #pragma endregion
}
