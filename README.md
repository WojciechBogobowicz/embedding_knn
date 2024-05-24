<!-- no toc -->
# TOC 
- [TOC](#toc)
- [Project overview](#project-overview)
  - [Objective](#objective)
  - [Advantages](#advantages)
  - [Applications](#applications)
  - [Key features](#key-features)
- [Environment setup](#environment-setup)
  - [On windows:](#on-windows)
  - [On Linux:](#on-linux)



# Project overview

## Objective
The primary objective of this project is to develop a robust image classification system leveraging the power of pre-trained deep learning models for feature extraction and a K-Nearest Neighbors (KNN) classifier for image classification. This approach provides a balance between computational efficiency and classification performance, making it suitable for various practical applications.

## Advantages
- **Computational Efficiency:** Training only the KNN classifier is computationally less intensive compared to fine-tuning the entire deep learning model.
- **Resource Efficiency:** This approach requires fewer computational resources and memory, making it suitable for environments with limited resources.
- **Simplicity and Rapid Prototyping:** The methodology allows for quick prototyping and deployment of image classification systems.
- **Effective Feature Utilization:** By leveraging pre-trained models, the system can utilize high-quality features learned from large datasets, enhancing classification performance.

## Applications
 - **Small to Medium Sized Datasets:** Ideal for scenarios with limited labeled data, where fine-tuning a deep learning model may lead to overfitting.
- **Resource-Constrained Environments:** Suitable for applications on edge devices or systems with limited computational power.
- **Prototype and Baseline Models:** Useful for quickly developing baseline models to evaluate the feasibility of using deep learning features for specific tasks.

## Key features
 - **Small Cost of Model Saving:** Only the fitted KNN classifier is stored locally, while the pre-trained deep learning model is accessed from TensorFlow Cloud, significantly reducing storage requirements.
- **Supported Pretrained Models:** Includes MobileNetV3Small, MobileNetV3Large, ResNet50, and VGG16, allowing users to choose the model that best fits their needs.
- **Supports custom model base** You can use your own pretrained model to extract embeddings.
- **Flexible Input Handling:** Supports both path strings and image arrays as input, making the model versatile and easy to integrate into different workflows.

# Environment setup
This repo contains setup folder with scripts thats allows you to create project environment on windows as well as on linux with minimal effort. Each script will create environment and install essential packages to run model, and after that ask you if you want also install packages required to run demo too.

## On windows:
Create environment:
```cmd
setup\environment_windows.bat
```
Use Environment:
```cmd
env\Scripts\activate
```

## On Linux:
Create environment:
```cmd
chmod +x setup/environment_linux.sh
./setup/environment_linux.sh
```
Use Environment:
```cmd
source env/bin/activate
```
