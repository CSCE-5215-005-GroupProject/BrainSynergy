# BrainSynergy: A Collaborative Kaggle Quest for Brain Tumor Identification

## Overview

This project focuses on the classification of MRI brain scans into one of four categories [Glioma Tumor, Meningioma Tumor, Normal (healthy brain), Pituitary Tumor]. As part of a group effort, each team member is developing individual models to tackle this classification problem. The project leverages an internal Kaggle competition to compare the performance of these models objectively.

## Dataset

The dataset comprises MRI brain scans, presented in a variety of angles and perspectives. Each scan falls into one of four distinct categories, making this a multi-class classification problem. The scans vary in terms of the visible brain regions and include both top-down and profile views. Each image is 256x256. There are 901 instances of class Glioma Tumor, 913 instances of class Meningioma Tumor, 438 instances of class Normal, and 844 instances of class Pituitary Tumor. The link to the dataset can be found on Kaggle here:  https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256

## Dataset Preparation

The dataset consists of MRI brain scans, each classified into one of four distinct categories. The preparation of this dataset involves the following steps:

### Step 1: Reserve Anonymous Kaggle Test Set

- 20% of the total images (3,096) are reserved for the Kaggle competition.
- This results in 619 images set aside for unbiased testing.

### Step 2: Dataset Split for Model Development

- The remaining 80% of the images are available for training and validation purposes.
- Each team member may choose their method of further splitting this subset for their model development.

## Team Members and Their Contributions

Below are the team members involved in this project, each with a subsection dedicated to their specific model and data plan.

### Joseph Caldwell (josephcaldwell@my.unt.edu)

As a part of the BrainSynergy Project, my primary focus has been on developing an innovative approach that combines a Generative Adversarial Network (GAN) with a repurposed discriminator and an ensemble of classifiers. 

The GAN's discriminator, originally trained to distinguish between real and generated MRI brain scans, is adapted as a powerful feature extractor.

The extracted features are then fed into two distinct classifiers: a Random Forest and a simple Neural Network. 

These classifiers, each bringing their unique strengths to the table, predict the presence and type of brain tumors in the scans. 

To optimize performance, the predictions from both classifiers are ensembled through a majority voting system, ensuring a robust and accurate diagnosis. This model not only leverages the generative power of GANs but also harnesses traditional machine learning and neural network techniques, creating a hybrid solution tailored for medical image analysis.


#### Data Plan

- **Step 1: Reserve Anonymous Kaggle Test Set**
  - 20% of 3,096 total images: 619 images (Reserved for Kaggle competition)
- **Step 2: Remaining Dataset**
  - 80% of 3,096 total images: 2,477 images (For GAN and classifier)
- **Step 3: GAN Training Subset**
  - 70% of 2,477: 1,734 images
  - After a single horizontal flip: 3,468 images for GAN training
- **Step 4: Classifier Training + Testing**
  - 30% of 2,477: 743 images
  - Training: 80% of 743 = 594 images
  - Testing: 20% of 743 = 149 images
- **Data Plan Summary**
  - Anonymous Kaggle Test Set: 619 images
  - GAN Training: 3,468 images (after a single data augmentation)
  - Classifier Training: 594 images
  - Classifier Testing: 149 images
 
### GAN Architecture Summary

#### Generator

The generator in this GAN is designed to upscale a noise input into a 256x256 grayscale image. Its architecture is as follows:

- **Initial Dense Layer**: This dense layer takes a noise vector of dimension `noise_dim` and transforms it into an 8x8x1024 tensor. This layer does not use bias and is followed by batch normalization and a LeakyReLU activation function.
- **Reshape**: The output from the dense layer is reshaped into a 3D tensor of shape 8x8x1024.
- **Deconvolutional Layers**: These layers progressively upscale the image's resolution while reducing its depth:
  - First layer upscales to 16x16 with 512 filters.
  - Second layer upscales to 32x32 with 256 filters.
  - Third layer upscales to 64x64 with 128 filters.
  - Fourth layer upscales to 128x128 with 64 filters.
- **Final Layer**: The final deconvolutional layer upscales to the target size of 256x256 with a single filter for grayscale output. It uses a 'tanh' activation function.
- Each deconvolutional layer uses a 5x5 kernel, a stride of 2x2, batch normalization (except the last layer), and LeakyReLU activation (except the last layer).

#### Discriminator

The discriminator's role is to classify images as real or generated. Its architecture is:

- **Input Layer**: Accepts 256x256x1 grayscale images (for RGB images, this would be 256x256x3).
- **Convolutional Layers**: These layers progressively downsample the image:
  - First layer downsamples to 128x128 with 64 filters, followed by dropout to prevent overfitting.
  - Second layer downsamples to 64x64 with 128 filters, followed by dropout to prevent overfitting.
  - Third layer downsamples to 32x32 with 256 filters, followed by batch normalization for more stable outputs. It is these outputs that will be used as features for the classification model.
  - Fourth layer downsamples to 16x16 with 512 filters, followed by dropout to prevent overfitting.
- **Flatten and Dense Layer**: The final output is flattened and passed through a dense layer with a single unit for binary classification.
- Each convolutional layer uses a 5x5 kernel, a stride of 2x2.

#### GAN Summary

This GAN architecture is tailored for processing MRI brain scans, with the generator creating realistic images from random noise and the discriminator distinguishing between real and generated images. The use of batch normalization and LeakyReLU activation functions contributes to the model's stability and efficiency during training.


#### GAN Development

- **Training Process**: 
  - Employed TensorFlow and Keras for model implementation and training.
  - Used Google Colab's GPU resources for efficient training.
  - Integrated checkpointing for managing long training sessions and resuming training as needed.
- **Performance Monitoring**: 
  - Tracked discriminator's accuracy and losses to gauge the GAN's training progress.
  - Regularly saved generated images to visually assess the GAN's output quality.
  - Trained for 970 epochs on a T4 GPU

#### Random Forest Classifier

- **Configuration:**
  - Utilized a Random Forest Classifier with 200 trees.
  - Applied default Scikit-Learn settings for other parameters.
 
- **Role in the Model:**
  - Key component of the ensemble model, providing robustness and handling complex data patterns.
  - Complements the Neural Network in feature interpretation and classification.

#### Neural Network Classifier

- **Architecture:**
  - A sequential model starting with a Flatten layer to convert 3D features into 1D.
  - First hidden layer: 128 neurons, ReLU activation.
  - Second hidden layer: 64 neurons, ReLU activation.
  - Output layer: 4 neurons (one for each class), Softmax activation.
    
- **Role in the Model:**
  - Key component of the ensemble model, providing robustness and handling complex data patterns.
  - Complements the Random Forest in feature interpretation and classification.


#### Challenges and Learnings

- Dealt with the nuances of GAN training, particularly balancing the generator and discriminator training.
- Learned to implement various TensorFlow functionalities for model checkpointing and resuming training.
- Ran into trouble installing TensorFlow locally, but given the complexity of the model, had to buy some cloud compute anyway, so did all training in Google Colab
 

#### Code and Resources

- The code for the GAN, along with documentation, is available in our GitHub repository.
- Training scripts and setup instructions are provided for replicating the model training process.



### Sina Montazeri (SinaMontazeri@my.unt.edu)

(Description of their model, data plan, and contributions.)

### Sonia Afrasiabian (SoniaAfrasiabian@my.unt.edu)

(Description of their model, data plan, and contributions.)

### Piyush Deepak (PiyushHemnani@my.unt.edu )

(Continue listing team members and their contributions.)

## Evaluation Metrics

Models are evaluated based on accuracy, precision, recall, F1 score, and other relevant metrics. The internal Kaggle competition setup allows for a fair and objective comparison of model performances.

## Usage

Instructions on how to set up the environment, train models, and evaluate them can be found in the respective model directories.

### Prerequisites

- TensorFlow
- Keras
- NumPy
- Pandas
- (Any other necessary libraries or tools)

### Installation and Setup

(Provide detailed instructions on how to install and set up the project, including any necessary virtual environments or dependencies.)

### Running the Models

(Provide step-by-step instructions on how to run the models, including any necessary commands or scripts.)


## Acknowledgements

(Any acknowledgments to data providers, supporting organizations, or individuals who have contributed to the project.)

