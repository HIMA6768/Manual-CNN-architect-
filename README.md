# 🧠 Manual-CNN: Convolutional Neural Network from Scratch
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-blueviolet?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Vision-green?style=for-the-badge)
![Neural Network](https://img.shields.io/badge/Neural%20Network-ANN-red?style=for-the-badge)

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

# 🧠 PureConv-Engine: Manual CNN from Scratch
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/library-NumPy-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular and mathematically grounded implementation of **Convolutional Neural Networks (CNN)** built entirely from scratch using **NumPy**. This project strips away high-level frameworks like TensorFlow or PyTorch to reveal the raw linear algebra and sliding-window logic that powers Computer Vision.

---

## 🚀 Features

- **Multi-Channel RGB Convolution:** Processes 3-channel image data with manual kernel integration.
- **Dynamic Auto-Padding:** Automatically calculates and applies padding based on kernel size.
- **Non-Linear Activation:** Integrated **ReLU** (Rectified Linear Unit) to filter feature maps.
- **Flexible Pooling:** Support for **Max**, **Average**, and **Min Pooling** layers.
- **Feature Flattening:** Seamlessly converts 2D feature maps into 1D vectors for Neural Network compatibility.
- **Custom Kernel Library:** Includes `Emboss`, `Sobel (Edges)`, `Sharpen`, and `Blur` filters.

---

## 🛠️ Mathematical Foundations

The engine calculates required padding and output dimensions using these standard CNN formulas:

### 1. Required Padding ($P$)
To ensure the output size remains consistent with the input (Same Padding), the required padding is calculated as:
$$P = \frac{F - 1}{2}$$
**(Where $F$ is the Filter/Kernel size)**

### 2. Output Dimension ($O$)
The spatial dimension of the output map is determined by:
$$O = \frac{I - F + 2P}{S} + 1$$
**(Where $I$: Input size, $S$: Stride)**

---

## 📂 Project Structure

```text
├── cnn.py            # Core Logic: Convolution, Pooling, ReLU, Flatten
├── kernels.py        # Library of Filters (Emboss, Sobel, Sharpen, etc.)
├── sampleinput.py    # Sample 3D Image Matrices
└── testcode.py       # Pipeline Execution & Orchestration
```
## 🧪 Installation & Usage
Copy and paste the following command into your terminal to clone, install, and run the project:
```
git clone [https://github.com/](https://github.com/)HIMA6768]/pureconv-engine.git && cd pureconv-engine && pip install numpy && python main.py

```
##🤝 Contributing
Contributions are welcome! If you want to add Backpropagation or Softmax layers, feel free to fork this repo and submit a pull request

## Developed with ❤️ by Himadri Ghosh
