# Image_Transfer_Model

This project is an initial attempt for exploring the basics of model training and parameter tuning for deep-learning application. The outcome generated is not expected to be compared to online image style transfer tools.

Fundamental Features:
Uses VGG19 pretrained on ImageNet to extract features.
Combines content and style representations using content loss and Gram matrix-based style loss.
Supports batch extraction of features from multiple style images.
Applies Adam Optimizer to minimise total loss by combining and style (difference between Gram matrices of the generated and style images) and content loss (difference between features of the generated and content image) to optimise the pixels of the generated image.
Averages style features to produce a more generalized artistic style.

Requirements:
Python 3.8+
PyTorch
torchvision
Pillow
matplotlib

How to run:
1. Place your content image in data/ (e.g., data/content_1.jpg).
2. Place style images in a folder like data/trainA/.
3. Run the main script

main.py is responsible for deploying the model by processing the input image.

Notes
Make sure CUDA is available if using a GPU. Training on CPU may be significantly slower. Training steps and style weight can be adjusted in train() method.
