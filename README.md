# 🧠 Manual-CNN: Convolutional Neural Network from Scratch
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/library-NumPy-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, modular, and mathematically grounded implementation of **Convolutional Neural Networks (CNN)** built entirely from scratch using **NumPy**. This project demonstrates the core mechanics of computer vision—from sliding window convolutions to feature map flattening—without relying on high-level frameworks like TensorFlow or PyTorch.

---

## 🚀 Features

- **Multi-Channel RGB Convolution:** Handles 3-channel image data with custom kernel integration.
- **Dynamic Padding & Stride:** Robust `addpadding` logic and stride-aware output dimension calculation.
- **Activation Functions:** Integrated **ReLU** (Rectified Linear Unit) for non-linearity.
- **Pooling Layers:** Support for **Max Pooling**, **Average Pooling**, and **Min Pooling** to reduce spatial dimensions.
- **Feature Flattening:** Seamlessly converts 2D feature maps into 1D vectors for Dense Neural Network compatibility.
- **Custom Kernel Support:** Easily switch between `Emboss`, `Sobel (Edges)`, `Sharpen`, and `Blur` filters.

---

## 🛠️ Mathematical Foundation

The core of this implementation follows the standard CNN spatial dimension formula:

$$O = \frac{I - K + 2P}{S} + 1$$

Where:
- $I$: Input size
- $K$: Kernel/Filter size
- $P$: Padding
- $S$: Stride

---

## 📂 Project Structure

```text
├── cnn.py            # Core logic: Convolution, Pooling, ReLU, Flatten
├── kernels.py        # Library of image processing filters (Emboss, Sobel, etc.)
├── sampleinput.py    # Sample image matrices for testing
└── testcode.py           # Pipeline execution and orchestration
```
## 🧪 Installation & Usage
Copy and paste the following command into your terminal to clone, install, and run the project:
```
git clone [https://github.com/](https://github.com/)HIMA6768]/pureconv-engine.git && cd pureconv-engine && pip install numpy && python main.py

```
##🤝 Contributing
Contributions are welcome! If you want to add Backpropagation or Softmax layers, feel free to fork this repo and submit a pull request

## Developed with ❤️ by Himadri Ghosh
