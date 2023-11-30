# BrainSynergy: Advanced Brain Tumor Classification Using GAN and Ensemble Classifiers

## Overview
This project is part of the BrainSynergy initiative, focusing on the development of a sophisticated machine learning model for classifying brain tumors using MRI scans. The approach involves a Generative Adversarial Network (GAN) for image generation and feature extraction, followed by an ensemble of classifiers for tumor identification.

## Repository Structure
- `datasetPrepJoe.ipynb`: Notebook for data preparation and augmentation, setting up datasets for GAN and classifier training/testing.
- `GAN_Training.ipynb`: Notebook dedicated to the training of the GAN model.
- `GAN_Classifier.ipynb`: Notebook for training classifiers using features extracted from the GAN's discriminator and generating final predictions.

## Dataset Preparation
The dataset consists of 3,096 MRI scans, further processed for model training:
- **Data Augmentation**: Implemented horizontal flip augmentation to increase the diversity of the GAN training dataset.
- **Data Splitting**: Separated the data into different sets for GAN training, classifier training, and classifier testing according to the specified data plan.

## GAN Model Training
The GAN model, comprising a generator and a discriminator, was trained to generate and distinguish MRI brain scans:
- **Training Environment**: Utilized Google Colab's T4 GPU for efficient training.
- **Epochs**: The model was trained for over 1,000 epochs, with a total training time of around 30 hours.
- **Checkpointing**: Implemented checkpointing for effective management of the lengthy training process.

## Classifier Training and Validation
Post GAN training, the discriminator was repurposed as a feature extractor for classifier training:
- **Classifiers**: Utilized a Random Forest Classifier and a simple Neural Network.
- **Ensemble Method**: Combined the predictions of both classifiers through majority voting for robust results.

## Usage
1. **Dataset Preparation**: Run `datasetPrepJoe.ipynb` to prepare and augment the data according to the project's data plan.
2. **GAN Training**: Execute `GAN_Training.ipynb` to train the GAN model. Adjust epochs and checkpointing as needed.
3. **Classifier Training and Testing**: Use `GAN_Classifier.ipynb` to train the classifiers with GAN-extracted features and generate final predictions.

## Future Directions and Improvements
- Explore enhanced data pipeline strategies incorporating cross-validation.
- Experiment with additional classification models and ensemble methods.

## References
1. TensorFlow Tutorials: [https://www.tensorflow.org/tutorials/](https://www.tensorflow.org/tutorials/)
2. Keras Examples: [https://keras.io/examples/vision/](https://keras.io/examples/vision/)
