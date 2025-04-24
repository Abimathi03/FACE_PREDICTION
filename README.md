# 🧠 Face Prediction using Masked Autoencoder

This project implements a **Face Mask Reconstruction** model using a **Convolutional Autoencoder**. It detects faces in an image, masks the lower half of the face, and reconstructs the masked area using a trained deep learning model. A custom loss function based on **Structural Similarity Index (SSIM)** is used to enhance the quality of reconstructions.

---

## 📑 Table of Contents

1. [✨ Features](#-features)  
2. [📦 Requirements](#-requirements)  
3. [⚙️ Installation](#-installation)  
4. [🚀 Usage of Intel OneAPI](#-usage-of-intel-oneapi)  
5. [🧬 Code Overview](#-code-overview)  
6. [📊 Visualization](#-visualization)  
7. [📄 License](#-license)  
8. [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Features

- 📷 Upload an image and detect faces using **MTCNN**
- 😷 Apply a mask to the lower half of the detected face
- 🧠 Reconstruct the masked region using a **Convolutional Autoencoder**
- 📊 Visualize the original, masked, and reconstructed images side-by-side

---

## 📦 Requirements

- Python 3.x  
- TensorFlow  
- OpenCV  
- Matplotlib  
- NumPy  
- dlib  
- facenet-pytorch  

---

## ⚙️ Installation

### 1. Clone the Repository

git clone https://github.com/Abimathi03/FACE_PREDICTION.git
cd face-mask-reconstruction

### 2. Install Dependencies

Install all required packages using pip:

pip install tensorflow opencv-python matplotlib numpy dlib facenet-pytorch

⚠️ Ensure you have CMake and Visual Studio Build Tools (for Windows) for installing dlib.

### 3. Download Pre-trained Models

Download the following file and place it in the project root:

- shape_predictor_68_face_landmarks.dat (for facial landmark detection)

You can get it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## 🚀 Usage of Intel OneAPI

Intel OneAPI is a toolkit for building optimized apps across CPUs and GPUs.

### Steps:

- Start a Jupyter Notebook or open a Google Colab session

- Run the main script to upload and process an image:

!python main.py

- Follow the on-screen prompts to upload your image. The system will:

- Detect the face

- Apply a lower-face mask

- Reconstruct the masked portion

- Display results

## 🧬 Code Overview
### 🔍 Face Detection
- Uses MTCNN to detect faces and facial landmarks.

### 🧼 Image Preprocessing
- Resize and normalize the input images.

### 🏗️ Autoencoder Architecture
- Encoder: Compresses input into a latent representation

- Decoder: Reconstructs the image from the latent vector

### 📐 Custom Loss Function
- Uses SSIM (Structural Similarity Index) to compare masked vs reconstructed images for better visual fidelity.

## 📊 Visualization

The system displays three outputs:

- Original image with detected face

- Masked face image (lower half masked)

- Reconstructed image using autoencoder

This side-by-side view makes it easy to assess model performance.

## 📄 License

This project is licensed under the MIT License.
See the LICENSE file for more information.

## 🙏 Acknowledgments

- MTCNN for face detection

- TensorFlow for deep learning models

- OpenCV for image processing utilities

- dlib for facial landmarks detection
