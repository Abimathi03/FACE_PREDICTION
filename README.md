Face Prediction

This project implements a face mask reconstruction model using a convolutional autoencoder. It applies a mask to the lower half of the detected face in an uploaded image and then reconstructs the masked area using a trained model. The model uses a custom loss function based on the Structural Similarity Index (SSIM) to enhance the quality of the reconstructed images.

Table of Contents

1)Features
2)Requirements
3)Installation
4)Usage of Intel OneAPI
5)License

Features

Upload an image and detect faces using MTCNN.
Apply a mask to the lower half of the detected face.
Reconstruct the masked area using a convolutional autoencoder.
Visualize the original masked image alongside the reconstructed image.

Requirements

1)Python 3.x
2)TensorFlow
3)OpenCV
4)Matplotlib
5)NumPy
6)dlib
7)facenet-pytorch

Installation

1)Clone the repository:
bash
Copy code
git clone https://github.com/Abimathi03/FACE_PREDICTION.git
cd face-mask-reconstruction

2)Install the required packages:
bash
Copy code
pip install -r requirements.txt

3)Download the pre-trained model for facial landmark detection (shape_predictor_68_face_landmarks.dat.bz2) and place it in the root directory.

Usage of Intel OneAPI 

Intel OneAPI is a software toolkit that enables developers to build and optimize applications for various hardware platforms, including CPUs, GPUs,
Start a Jupyter Notebook or Google Colab environment.
Run the main script to upload an image and process it:
python
Copy code
# Run the script
!python main.py
Follow the prompts to upload an image. The program will display the original image with the detected face, the masked image, and the reconstructed image.

Code Overview
Face Detection: Uses MTCNN for detecting faces and landmarks.
Image Preprocessing: Resizes and normalizes images for processing.
Autoencoder Architecture: Consists of an encoder that compresses the image and a decoder that reconstructs the image.
Custom Loss Function: Utilizes SSIM for better image quality during reconstruction.

Visualization

The code visualizes the masked image and the reconstructed output side by side for easy comparison.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

MTCNN for face detection.
TensorFlow for machine learning framework.
OpenCV for image processing.
