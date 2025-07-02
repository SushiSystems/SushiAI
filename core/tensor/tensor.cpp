/**************************************************************************/
/*  tensor.cpp                                                            */
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

#include <memory>
#include <random>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include "tensor.h"

namespace SushiAI
{
    #pragma region Tensor Constructor

    Tensor::Tensor(const std::vector<int>& shape, float fill, bool requiresGrad) : shape(shape), requiresGradient(requiresGrad), totalSize(1)
    {
        for (int s : shape)
            totalSize *= s;

        data.resize(totalSize, fill);
        gradient.resize(totalSize, 0.0f);

        calculateStrides();
    }

    #pragma endregion

    #pragma region Tensor Functions

    std::shared_ptr<Tensor> Tensor::Zeros(const std::vector<int>& shape, bool requiresGrad)
    {
        return std::make_shared<Tensor>(shape, 0.0f, requiresGrad);
    }

    std::shared_ptr<Tensor> Tensor::Ones(const std::vector<int>& shape, bool requiresGrad)
    {
        return std::make_shared<Tensor>(shape, 1.0f, requiresGrad);
    }

    int Tensor::getFlatIndex(std::initializer_list<int> indices) const
    {
        assert(indices.size() == shape.size());

        int idx = 0, i = 0;
        for (int ind : indices)
        {
            assert(ind >= 0 && ind < shape[i]);

            idx += ind * strides[i];
            i++;
        }

        return idx;
    }

    float& Tensor::at(std::initializer_list<int> indices)
    {
        return data[getFlatIndex(indices)];
    }

    const float& Tensor::at(std::initializer_list<int> indices) const
    {
        return data[getFlatIndex(indices)];
    }

    void Tensor::calculateStrides()
    {
        strides.resize(shape.size());

        int stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    void Tensor::reshape(const std::vector<int>& newShape)
    {
        int newSize = 1;

        for (int dim : newShape)
            newSize *= dim;

        assert(newSize == totalSize);

        shape = newShape;
        calculateStrides();
    }
    
    void Tensor::print(const std::string& name) const
    {
        std::cout << "====== Tensor Debug ======\n";

        if (!name.empty())
            std::cout << "Name     : " << name << "\n";

        std::cout << "Shape    : [";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i];

            if (i < shape.size() - 1) 
                std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "Values   : [";
        if (shape.size() == 1) 
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                std::cout << data[i];
                if (i < data.size() - 1) 
                    std::cout << ", ";
            }
        }
        else if (shape.size() == 2) 
        {
            std::cout << "\n";
            for (int i = 0; i < shape[0]; ++i)
            {
                std::cout << "  [";
                for (int j = 0; j < shape[1]; ++j)
                {
                    std::cout << at({ i, j });
                    if (j < shape[1] - 1) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
        else 
        {
            std::cout << "(printing not implemented for shape > 2)";
        }
        std::cout << "]\n";

        if (!gradient.empty())
        {
            std::cout << "Gradient : [";
            if (shape.size() == 1) 
            {
                for (size_t i = 0; i < gradient.size(); ++i)
                {
                    std::cout << gradient[i];
                    if (i < gradient.size() - 1) std::cout << ", ";
                }
            }
            else if (shape.size() == 2) 
            {
                std::cout << "\n";
                for (int i = 0; i < shape[0]; ++i)
                {
                    std::cout << "  [";
                    for (int j = 0; j < shape[1]; ++j)
                    {
                        std::cout << gradient[i * shape[1] + j];
                        if (j < shape[1] - 1) std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
            }
            else 
            {
                std::cout << "(printing not implemented for shape > 2)";
            }
            std::cout << "]\n";
        }

        std::cout << "===========================\n";
    }

    #pragma endregion

    #pragma region Computation Graph

    std::vector<Tensor*> Tensor::topologicalSort() const
    {
        std::vector<Tensor*> topo;
        std::unordered_set<Tensor*> visited;

        std::function<void(const Tensor*)> dfs = [&](const Tensor* node)
        {
            if (!node || visited.count((Tensor*)node))
                return;

            visited.insert((Tensor*)node);

            for (auto& p : node -> parents)
                dfs(p.get());

            topo.push_back((Tensor*)node);
        };

        dfs(this);

        return topo;
    }

    #pragma region Backpropagation Mechanism, REFACTORING NEEDED

    void Tensor::backward(bool retainGraph, bool clearExisting)
    {
        std::vector<float> seed(totalSize, 0.0f);
        seed[0] = 1.0f;

        backward(seed, retainGraph, clearExisting);
    }

    void Tensor::backward(const std::vector<float>& seed, bool retainGraph, bool clearExisting)
    {
        auto topo = topologicalSort();

        if (clearExisting)
            for (auto* n : topo)
                n -> gradient.assign(n -> totalSize, 0.0f);

        assert((int)seed.size() == this -> totalSize);

        this -> gradient = seed;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            if ((*it) -> gradientFunction)
                (*it) -> gradientFunction();
        }

        if (!retainGraph)
        {
            for (auto* n : topo)
            {
                n -> gradientFunction = nullptr;
                n -> parents.clear();
            }
        }
    }

    #pragma endregion

    #pragma endregion
}
