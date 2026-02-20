# ML-Engineer-Technical-Test-AI_Monk

# Multilabel Image Classification

This repository contains the solution for a multilabel image classification problem. The goal is to predict the presence of four distinct attributes within a given image. The dataset presents specific real-world challenges, including missing annotations ("NA") and severe class imbalances.

## 📌 Deliverables Completed
- [x] **Training Code:** Fine-tunes a deep learning model to output weights.
- [x] **Loss Curve Plot:** Tracks training loss over iterations (`Aimonk_multilabel_problem`).
- [x] **Inference Code:** Accepts an image and outputs the predicted list of present attributes.

## 🧠 Architecture & Approach

* **Framework:** PyTorch 
* **Base Model:** `ResNet18` 
* **Strategy:** Fine-tuning on top of ImageNet trained weights. The final fully connected layer is replaced with a 4-node linear layer to predict the four distinct attributes.

## 🛠 Handling Dataset Challenges

### 1. Missing Annotations ("NA")
Several images lack ground-truth data for specific attributes, denoted as "NA" in the `labels.txt` file. 
* **Solution:** Missing labels are converted to `-1.0`. During the forward pass, the loss is calculated using `BCEWithLogitsLoss(reduction='none')`. A mask is applied to zero-out the loss for missing attributes, ensuring we do not ignore the image completely.

### 2. Class Imbalance
The dataset is skewed, with a huge difference in the number of images for each attribute.
* **Solution implemented:** Dynamic class weighting. Before training, the script calculates the ratio of negative to positive samples for each attribute. These ratios are passed into the `pos_weight` parameter of the PyTorch Loss function, penalizing the model more heavily for missing minority classes.

## 🚀 Future Improvements & Thoughts
Due to time constraints, the following techniques were not fully implemented but would be the immediate next steps to improve model robustness:

1. **Focal Loss:** While `pos_weight` helps with imbalance, implementing Focal Loss would force the model to focus explicitly on the "hard-to-classify" examples.
2. **Advanced Augmentations:** Currently using `RandomHorizontalFlip`. Adding `ColorJitter`, `RandomRotation`, or `Cutout` would prevent overfitting.
3. **Multilabel Stratified Split:** Implementing Iterative Stratification to ensure that the distribution of all 4 attributes remains consistent across training and validation splits.

## 💻 How to Run

### Setup
Clone the repository:
```bash
git clone [https://github.com/kabirkohli123/multilabel-classification.git](https://github.com/kabirkohli123/ML-Engineer-Technical-Test-AI_Monk.git)
cd multilabel-classification
