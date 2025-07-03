/**************************************************************************/
/*  sequential.cpp                                                        */
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

#include <fstream>
#include <filesystem>

#include "sequential.h"

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

namespace SushiAI 
{
    #pragma region Sequential Constructor

    Sequential::Sequential(const std::vector<std::shared_ptr<Layer>>& layers) : layers(layers) { }
    
    #pragma endregion

    #pragma region Sequential Functions

    std::shared_ptr<Tensor> Sequential::forward(const std::shared_ptr<Tensor>& input, bool training)
    {
        auto out = input;

        for (auto& layer : layers)
            out = layer -> forward(out, training);

        return out;
    }

    std::vector<std::shared_ptr<Tensor>> Sequential::parameters() const
    {
        std::vector<std::shared_ptr<Tensor>> params;

        for (auto& layer : layers) 
        {
            auto p = layer -> parameters();
            params.insert(params.end(), p.begin(), p.end());
        }

        return params;
    }

    void Sequential::add(const std::shared_ptr<Layer>& layer)
    {
        layers.push_back(layer);
    }

    void Sequential::remove(size_t index)
    {
        if (index < layers.size())
            layers.erase(layers.begin() + index);
        else
            throw std::out_of_range("Sequential::remove(): index out of range");
    }

    void Sequential::saveModel(const std::string& filename) const
    {
        std::string dir = std::string(PROJECT_ROOT_DIR) + "/models";
        std::filesystem::create_directories(dir);
        std::string path = dir + "/" + filename;

        std::ofstream file(path);

        if (!file.is_open())
        {
            std::cerr << "Cannot open file to save model: " << path << "\n";
            return;
        }

        for (const auto& layer : layers)
        {
            for (const auto& param : layer -> parameters())
            {
                const auto& data = param -> getData();

                for (float val : data)
                    file << val << " ";

                file << "\n";
            }
        }

        file.close();
        std::cout << "Model saved to: " << path << "\n";
    }

    void Sequential::loadModel(const std::string& filename)
    {
        std::string dir = std::string(PROJECT_ROOT_DIR) + "/models";
        std::filesystem::create_directories(dir);
        std::string path = dir + "/" + filename;
        
        std::ifstream file(path);

        if (!file.is_open())
        {
            std::cerr << "Cannot open file to load model: " << path << "\n";
            return;
        }

        for (auto& layer : layers)
        {
            for (const auto& param : layer -> parameters())
            {
                auto& data = param -> getData();

                for (float& val : data)
                    file >> val;
            }
        }

        file.close();
        std::cout << "Model loaded from: " << path << "\n\n";
    }

    #pragma endregion
}
