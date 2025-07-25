# SushiAI

**Physics-Informed Deep Learning Framework for Simulations.**

## About Project

**SushiAI** is a lightweight, high-performance, modular deep learning framework written in modern C++. Built with simulation in mind, SushiAI excels at accelerating physics-based computations using neural networks. It supports Physics-Informed Neural Networks (PINNs), modular tensor operations, and is designed to run efficiently even on low-end machines.

**Vision:** "To deliver real-time performance in simulations without cutting corners on the laws of physics, powered by deep learning."

**Mission:** "To run Sushi Systems Virtual World project, Project: RL, on bit-level hardware with the highest possible performance."

---

## Features

- Minimal and fast tensor engine
- Autograd (automatic differentiation) support
- Built-in optimizers (SGD, Adam, etc.)
- Modular activation functions (ReLU, Tanh, Sigmoid, Softmax, and more...)
- CPU support (x86), CUDA in development
- Saving & Loading a trained model in SushiAI
- Physics-Informed Neural Networks (PINNs) (WIP)
- Graph-based Neural Networks (GNNs) (WIP)
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
>
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

---

## Documentation

- [API Reference (WIP)](docs/)
- [Examples](examples/)
- [Design Notes](docs/design.md)

---

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

## 🧪 Examples

- Pure MLP training
- Solving PDEs like Navier-Stokes, Poison or heat transfer simulations with PINNs
- Unity3D integration (planned)

> See the `examples/` folder for detailed usage.

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

*This project is all about using AI to handle complex physical computations in real-time, without compromising the integrity of real-world physics.
If you’ve got any questions or just want to share your thoughts, feel free to reach out through the links in the License and Contact section. I’ll get back to you as soon as I can. Thanks for stopping by and take care!*

