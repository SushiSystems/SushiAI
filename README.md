# SushiAI

**Lightweight Deep Learning Framework for Simulations.**

---

## About Project

**SushiAI** is a lightweight and modular deep learning framework created as the main AI platform for Sushi Systems projects. It is written in C++ and designed to run efficiently on both mobile devices and desktop computers.

The focus of SushiAI is on reinforcement learning and on using artificial intelligence to simulate natural sciences on a computer. In practice, it is meant to provide fast training and inference for AI agents in real-time games, while in the long run it will also serve as a foundation for scientific simulations.

At the moment it supports simple MLP models. Work on reinforcement learning is actively in progress, while support for PINN and GNN is still at an early draft stage and planned as part of longer-term research.

---

## Features

- Minimal and fast tensor engine
- Autograd (automatic differentiation) support
- Built-in optimizers (SGD, Adam, etc.)
- Modular activation functions (ReLU, Tanh, Sigmoid, Softmax, and more...)
- CPU support (x86), CUDA in development
- Saving & Loading a trained model in SushiAI
- Physics-Informed Neural Networks (PINNs) (planned)
- Graph-based Neural Networks (GNNs) (planned)
- Python and Unity3D frontends (planned)

---

## Installation

### Build from Source

```bash
git clone https://github.com/YOUR_USERNAME/SushiAI.git
cd SushiAI
mkdir build && cd build
cmake ..
make -j4
```

> **Requirements:**
> - CMake ≥ 3.14
> - C++14-compliant compiler (GCC or Clang)

---

## Getting Started

Below is a minimal example for training a simple Multi-Layer Perceptron (MLP) model to predict constant values:

```cpp
#include "tensor.h"
#include "ops.h"
#include "loss.h"
#include "layer.h"
#include "optimizer.h"
#include "sequential.h"
#include "initializer.h"

using namespace SushiAI;

int main() 
{
    auto model = std::make_shared<Sequential>();
    model -> add(std::make_shared<Linear>(2, 16, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));
    model -> add(std::make_shared<ReLU>());
    model -> add(std::make_shared<Linear>(16, 1, std::make_shared<XavierUniform>(), std::make_shared<XavierUniform>()));

    auto input = Tensor::Ones({100, 2});
    auto target = Tensor::Zeros({100, 1});

    auto lossFunction = std::make_shared<MSELoss>();
    auto optimizer = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 0; epoch < 100; ++epoch)
    {
        auto prediction = model -> forward(input);
        auto loss = lossFunction -> forward(prediction, target);

        loss -> backward();
        optimizer -> step(model -> parameters());
        optimizer -> zeroGradient(model -> parameters());

        std::cout << "Epoch " << epoch << " Loss: " << loss -> getData()[0] << std::endl;
    }

    model -> saveModel("mlp.sushi");

    return 0;
}
```

## Project Structure

```bash
SushiAI/
└── core/ 
    ├── loss/           # Loss functions
    ├── neuralnetwork/  # Linear (Dense), Sequential, Activations, etc.
    ├── operations/     # add, matmul, element-wise ReLU, Tanh, etc.
    ├── optimizer/      # Adam, SGD, etc.
    └── tensor/         # Tensor engine
└── docs/               # Documents
    └── versions/       # Version log
├── models/             # SushiAI Model save & load directory (.sushi)
└── sushiBLAS/          # SushiBLAS Linear Algebra library
    └── core/           # BLAS Backend
```

---

## Examples

- MLP training

---

## Roadmap

| Feature                   | Status     |
| ------------------------- | ---------- |
| Core Tensor Engine        | ✅ Done    |
| Basic Autograd            | ✅ Done    |
| Basic NN Structure        | ✅ Done    |
| Optimizers (Adam, SGD)    | ✅ Done    |
| BLAS Backend (SushiBLAS)  | 🔜 In Dev  |
| Dynamic Computation Graph | 🔜 Planned |
| PINN & GNN Support        | 🔜 Planned |
| CUDA Acceleration         | 🔜 Planned |
| Python API (PySushi)      | 🔜 Planned |
| Unity Runtime Integration | 🔜 Planned |

---

## Contributing

We welcome all contributions to SushiAI.

- Fork the repo and create your feature branch.
- Follow C++ style used in the core.
- Write tests if applicable.
- Submit a pull request.

> Please read our [Contributing Guidelines](CONTRIBUTING.md) before contributing.

---

## License

SushiAI is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author & Contact

* **Author:** Mustafa Garip
* **E-Mail:** [mustafagarip@sushisystems.io](mailto:mustafagarip@sushisystems.io)
* **GitHub:** [https://github.com/sushimg/SushiAI](https://github.com/sushimg/SushiAI)
* **LinkedIn:** [https://www.linkedin.com/in/mustafgarip](https://www.linkedin.com/in/mustafgarip)

>*This project is all about using AI to handle complex physical computations in real-time, without compromising the integrity of real-world physics.*
>*If you’ve got any questions or just want to share your thoughts, feel free to reach out through the links in the License and Contact section. I’ll get back to you as soon as I can. Thanks for stopping by and take care!*

