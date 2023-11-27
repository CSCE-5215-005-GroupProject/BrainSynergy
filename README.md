# MRI Brain Scan Classification Project

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

(Description of their model, data plan, and contributions.)

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

