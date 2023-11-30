# Vision transformer

# The Vision Transformer (ViT) 
ViT is a novel architecture in the field of computer vision and it is inspired by the impact of transformers in natural language processing. Unlike traditional approaches that typically rely on convolutional neural networks (CNNs), ViT shows that a pure transformer applied directly to sequences of image patches can achieve impressive results in image classification tasks, especially when pre-trained on substantial datasets​​.[1]

# Architecture
At its core, ViT adopts the transformer architecture with minimal modifications for compatibility with image data. Images are first split into fixed-size patches, which are then linearly embedded, akin to token embeddings in NLP. These patch embeddings are treated as input sequences for the transformer. Each patch is processed as if it were a word token in a sentence, highlighting the model's versatility across different domains​​.[1]

# Components of ViT:
Patch and Position Embeddings: The initial step involves reshaping the image into a sequence of flattened 2D patches. These patches are then mapped to a constant latent vector size, a process known as patch embedding. Additionally, ViT incorporates learnable 1D position embeddings, added to the patch embeddings to retain positional information within the image​​.[1]

## Transformer Encoder: 
The encoder consists of alternating layers of multi-headed self-attention (MSA) and multi-layer perceptrons (MLPs), with layernorm applied before each block and residual connections afterward. This structure is a direct adoption from NLP transformers​​.[1]

## Classification Token:
Similar to BERT's class token, ViT introduces a learnable embedding at the beginning of the patch sequence. The state of this token at the output of the transformer encoder is used for image classification, attached to either an MLP with one hidden layer during pre-training or a single linear layer during fine-tuning​​.[1]

## MLP Blocks: 
The MLP within the transformer encoder contains two layers with a GELU non-linearity​​. These blocks are interleaved with the multi-headed self-attention layers within the transformer encoder. 
* The primary function of the MLP blocks is to apply a non-linear transformation to the data. This is essential because the self-attention mechanism is inherently linear in nature. Without these non-linear transformations, the model would be unable to capture complex patterns in the data, regardless of the depth of the network.
* Adding MLP layers increases the depth of the network, which allows for a more complex and nuanced understanding of the input data
* The inclusion of MLP blocks adds flexibility to the network, enabling it to be applicable to a variety of tasks beyond just image classification.

MLP blocks in ViT contribute significantly to its ability to learn complex representations from image data. They complement the self-attention mechanism by adding necessary non-linear processing power, integrating features, and enhancing the model's depth and adaptability.

# Inductive Bias:
Unlike CNNs that inherently possess biases like translation equivariance and locality, ViT has much less image-specific inductive bias. The model learns spatial relations between patches from scratch, except during initial patch extraction and fine-tuning for position embeddings adjustment​​.[1]

# Training and Fine-tuning
ViT models are typically pre-trained on large datasets and fine-tuned for specific downstream tasks. The fine-tuning process involves removing the pre-trained prediction head and attaching a new feedforward layer corresponding to the number of classes in the downstream task. Notably, fine-tuning often occurs at a higher resolution than the pre-training, necessitating adjustments in the position embeddings​​.

# Optimizer:
For training, the Adam optimizer is utilized with specific parameters (β1 = 0.9, β2 = 0.999, and a high weight decay of 0.1), found effective for transferring models. The learning rate follows a linear warmup and decay pattern. During fine-tuning, stochastic gradient descent (SGD) with momentum is employed​​.

# Application in Image Classification
* Dataset Preparation: A custom BrainTumorDataset class is implemented to process and load images as per the requirements of the ViT model.

* Model Initialization: The ViTForImageClassification class encapsulates the ViT model with a classifier head tailored for the specific number of labels in the dataset.

* Training Loop: The model is trained using the Adam optimizer, with the training process including loss calculation, backpropagation, and optimization steps.

* Evaluation: A function evaluate_model is provided for assessing the model's performance on a validation dataset, calculating metrics such as loss and accuracy.

* Model Saving and Loading: The model's state is saved after each epoch, and a trained model can be loaded for further predictions or analysis.

# Conclusion
the Vision Transformer represents a significant shift in the approach to image classification, leveraging the power of transformer architectures without reliance on CNNs. The provided code exemplifies its practical implementation in a real-world scenario, showcasing its effectiveness and versatility.



[1] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. Dosovitskiy et. al. https://arxiv.org/abs/2010.11929
