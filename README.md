# Developing a Model Using Pretrained Deep Convolutional Neural Networks (DCNNs) for Detecting Cassava Leaf Diseases

## Background
Cassava is a crucial crop in many parts of the world, providing food for millions of people. However, cassava production faces significant threats from various diseases that can severely reduce yields. Early detection of these diseases is vital to ensure effective management and prevent widespread damage. This project focuses on detecting cassava leaf diseases using deep learning models, specifically leveraging pretrained Deep Convolutional Neural Networks (DCNNs).

## Objective
The primary objective of this project is to develop a model capable of accurately detecting cassava leaf diseases. By utilizing pretrained DCNN architectures like VGG16, VGG19, ResNet50, ResNet101, and XCeption, the goal is to compare their performance and identify the best model for this task. The experiment involves two key stages: transfer learning and fine-tuning.

## Dataset
The dataset used for this project contains 21,391 images of cassava leaves from [Kaggle](https://www.kaggle.com/nirmalsankalana/cassava-leaf-disease-classification). These images are classified into five categories:
- Healthy
- Cassava Mosaic Disease (CMD)
- Cassava Bacterial Blight (CBB)
- Cassava Green Mite (CGM)
- Cassava Brown Streak Disease (CBSD)

The images were taken using smartphone cameras and were publicly sourced. The dataset is split into:
- **80% for training**
- **10% for validation**
- **10% for testing**

Each image is resized to 224 x 224 pixels and represented in RGB format.

## Experiment Setup
The experiment evaluates several pretrained DCNN models, including:
- **VGG16**
- **VGG19**
- **ResNet50**
- **ResNet101**
- **XCeption**

### Key Parameters
- **Image resolution**: 224 x 224 pixels
- **Batch size**: 16
- **Learning rate**:
  - Transfer learning phase: 0.0001
  - Fine-tuning phase: 0.00001
- **Optimization algorithm**: Adam
- **Loss function**: Categorical Crossentropy
- **Early Stopping**: Applied with patience of 5 epochs to avoid overfitting.
  
Model checkpoints are used to save the best-performing weights during training. If model performance decreases during training, it is stopped early.

## Methodology

### 1. **Transfer Learning**
Pretrained models were utilized by freezing their initial layers, which are responsible for extracting basic features. The classification head was replaced with a custom fully connected layer that outputs predictions for the 5 classes. The initial layers remain frozen, while only the newly added layers are trained.

### 2. **Fine-Tuning**
In the fine-tuning stage, some of the deeper layers of the pretrained models are unfrozen, and their weights are updated along with the classification head to improve accuracy further. The learning rate is reduced to avoid large updates during this stage.


### Performance Evaluation
After training, the models were evaluated using the test dataset. Accuracy, confusion matrices, and classification reports were generated to assess their performance.

## Results
The following results were obtained:
<img width="957" alt="image" src="https://github.com/user-attachments/assets/acf0bb1a-0868-4b36-a35a-5bee04d12eb6">

- **VGG16** outperformed all other models in both transfer learning and fine-tuning stages.
- **VGG19** consistently provided the second-best performance, closely following VGG16.
  

## Conclusion
VGG16 proved to be the most effective model for detecting cassava leaf diseases, achieving the highest accuracy during both transfer learning and fine-tuning phases. This indicates that VGG16's architecture, coupled with transfer learning from ImageNet, provides robust feature extraction and classification capabilities for this task. VGG19 also demonstrated strong performance, making it a viable alternative.
