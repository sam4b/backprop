# Backprop

A small C++ library for training deep neural networks, with both a CPU-side training backend (the Eigen linear algebra library)
and a GPU-accelerated training backend (via CUDA, so NVIDIA GPUs only).

Examples can be found in ```examples/``` (MNIST, for now).

## Build instructions:
Install CUDA (even if you have no NVIDIA GPU, the code must be compiled)

Use CMake to build (Visual Studio has been used to test this, but the terminal should work too)

## Limitations:


## Future:
Adding more loss functions, and providing for specific per-layer activation functions.
A pybind11 binding to Python to allow the use of Jupyter notebooks.
Convolutional neural network support.
Saving and loading GPU models.

## Acknowledgements:
The libraries Eigen, nlohmann-json, cuBLAS, CUDA and rapidCSV were used in this project. 
The MNIST dataset is utilised.