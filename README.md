# BrainSynergy: A Collaborative Kaggle Quest for Brain Tumor Identification

## Overview

This project focuses on the classification of MRI brain scans into one of four categories [Glioma Tumor, Meningioma Tumor, Normal (healthy brain), Pituitary Tumor]. As part of a group effort, each team member is developing individual models to tackle this classification problem. The project leverages an internal Kaggle competition to compare the performance of these models objectively.

## Dataset

The dataset used in this project comprises MRI brain scans, each categorized into one of four distinct classes. This diversity in the dataset introduces a comprehensive multi-class classification challenge. Key features of the dataset include:

- **Variety of Angles and Perspectives**: The MRI scans present a range of views, including both top-down and profile perspectives.
- **Multi-Class Classification**: The dataset is structured for a multi-class classification problem with four distinct categories.
- **Class Distribution**:
  - **Glioma Tumor**: 901 instances
  - **Meningioma Tumor**: 913 instances
  - **Normal (Healthy Brain)**: 438 instances
  - **Pituitary Tumor**: 844 instances
- **Image Specifications**: Each scan is a 256x256 pixel image, offering detailed visualization for accurate classification.
- **Dataset Source**: The dataset is accessible on Kaggle and can be found at the following link: [Brain Tumors 256x256 on Kaggle](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256).
## Dataset Preparation

The dataset consists of MRI brain scans, each classified into one of four distinct categories. The preparation of this dataset involves the following steps:

### Step 1: Reserve Anonymous Kaggle Test Set (data/kaggle_solutionSet)

- 20% of the total images (3,096) are reserved for the Kaggle competition.
- This results in 619 images set aside for unbiased testing.

### Step 2: Dataset Split for Model Development (data/competition_data)

- The remaining 80% of the images are available for training and validation purposes.
- Each team member may choose their method of further splitting this subset for their model development.

These steps occur in the datasetPrep.ipynb file within the data folder. 

## Repository Structure

- `base_model`: Contains the base model (VGG16).
- `data`: 
  - Contains the original dataset, competition data, and kaggle_solutionSet.
  - `datasetPrep.ipynb` is used for preparing the competition dataset.
  - `sandbox.csv` and `solution.csv` for Kaggle competition setup.
- `joe_model`, `piyush_model`, `sina_model`, `sonia_model`: 
  - Individual directories for each team member.
  - Internal structures and hierarchies are determined by each member.

## Team Member Models

Each team member's directory (`joe_model`, `piyush_model`, `sina_model`, `sonia_model`) contains their individual model development and results. The specifics of each model, including architecture, data handling, training, and validation approaches, are documented within these directories.


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

#### Ensemble Method

- **Composition:**
  - The ensemble combines predictions from the Random Forest Classifier and the Neural Network Classifier.
- **Voting Mechanism:**
  - Majority voting system used to finalize the classification decision. Each classifier's prediction is considered a 'vote', and the class with the majority votes is selected as the final output.

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

## Model Overview:
The model leverages the EfficientNetB0 architecture, which is acclaimed for balancing model complexity with accuracy. Using ImageNet's extensive dataset, it's a solid foundation, able to recognize a broad range of features.  
 
 By employing transfer learning, the pre-trained model was fine-tuned to specialize in the identification of brain tumor images. We're using this approach to apply ImageNet's learning to our specific medical imaging task.  
 
The model was adapted to our unique dataset, comprising various brain tumor MRI images. The adaptability of EfficientNetB0 enables it to learn from our specific data, thereby enhancing its diagnostic accuracy. 
 
Objective: the project's objective is to harness the model's advanced feature recognition to improve the accuracy of diagnosing brain tumors,and  employ this model to detect and classify brain tumors.


## Model Architecture & Design:

- Foundation: I Utilized the EfficientNetB0 architecture, which was pre-trained on the ImageNet dataset. I Utilized the EfficientNetB0 architecture, which was pre-trained on the ImageNet dataset. In this model, all dimensions of depth, width, and resolution are uniformly scaled using a compound scaling method. 

- Customization: Input shape tailored to (224x224x3) to match our dataset, allowing the model to process our MRI images effectively. This size was chosen as a balance between detail retention and computational efficiency. 

- Data Augmentation: To prevent the model from overfitting and improves its ability to generalize to new, unseen data, I implemented an augmentation strategy with: 
Random horizontal and vertical flips. 
Random rotations of up to 20% of 360 degrees. 
Random zooming up to 20%. 
Random brightness changes, enhancing contrast by up to 20%. These augmentations were integrated using TensorFlow's Keras preprocessing layers. 

- Pooling and Classification Layers: After the base EfficientNetB0 layers, I used GlobalAveragePooling2D to compress the feature maps, reducing the model's complexity and computational load. The final layer is a dense layer with softmax activation, classifying into four categories: glioma, meningioma, notumor, and pituitary tumors. 

## Model Training & Validation:

- Dataset Management: Our dataset, comprising high-resolution brain MRI images, was systematically divided, reserving 80% for training and 20% for validation. This distribution was meticulously chosen to ensure a robust training set while retaining a representative validation set for accurate model performance assessment. 

- Training Strategy: The model was compiled with the Adam optimizer, Setting the Model to be Trainable: Initially, the base model, EfficientNetB0, was set to be not trainable (base_model.trainable = False) during the first phase of training. This was done to freeze the weights of the pre-trained network, so only the new layers I added (like the Dense output layer) would be trained.The model was trained over 100 epochs .

- Hyperparameters and Validation:I used a batch size of 32 to balance between computational efficiency and model performance.The validation split was set at 20% to ensure a sufficient amount of data for model evaluation while maximizing the data available for training. 

- Performance Metrics: befor Fine_ Tunning my model achieved an accuracy of : 0.85% on the training set and 0.77% on the validation set . 

### Fine-tuning in my project was implemented as follows:

- Enabling Fine-Tuning: After the initial training, the base model was set to be trainable (base_model.trainable = True). This step allows the weights in the pre-trained layers to be updated during training. 

- Selective Layer Training: I decided to fine-tune from a certain layer onwards (fine_tune_at = 100). This means that the layers before the 100th layer in the EfficientNetB0 model were kept frozen (layer.trainable = False for layers in base_model.layers[:fine_tune_at]), and the layers from the 100th onwards were set to be trainable. (which capture more specific features). 

- Hyperparameter Optimization and Re-compiling the Model: the learning rate, initially set to 1e-4, and the batch size of 32 were iteratively fine-tuned. This is important in fine-tuning because a lower learning rate ensures that the model does not deviate too much from the learned weights. 

- Continued Training: Finally, the model was trained for additional epochs (FINE_TUNE_EPOCHS = 60, 40). This continued training allowed the trainable layers of the base model to adjust their weights (fine-tune) to better fit the specific data of your project. 

## Performance Metrics: 
- befor Fine_ Tuning my model achieved an accuracy of : 0.85% on the training set and 0.77% on the validation set . 

- After Fine_ Tuning my model achieved an accuracy of : 0.97% on the training set and 0.91% on the validation set.


### Piyush Deepak (PiyushHemnani@my.unt.edu )

(Continue listing team members and their contributions.)

## Evaluation Metrics

Models are evaluated based on accuracy using an internal Kaggle competition setup allows for a fair and objective comparison of model performances. Additional metrics may be offered in each group members' respective directory. 

## Usage

Instructions on how to set up the environment, train models, and evaluate them can be found in the respective model directories.


