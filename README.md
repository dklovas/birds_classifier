# Bird Species Classification

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Conclusion](#conclusion)
4. [Technologies Used](#technologies-used)
5. [How to Use](#how-to-use)

## Overview

This project focuses on classifying bird species using the NABirds dataset, which contains 48,000 annotated images across 400 bird species found in North America. The task is to develop a model that can classify bird species with high accuracy, using techniques such as few-shot learning and pre-trained models with fine-tuning. The dataset provides both labeled and unlabeled images, with the goal of improving model performance through iterative labeling of the data.

In this project, we explore different methods for handling limited labeled data, build and evaluate two different classifiers, and use techniques such as transfer learning to improve classification accuracy. We also analyze model predictions using LIME (Local Interpretable Model-agnostic Explanations) to gain insights into how the model is making decisions.

## Objectives

The main objectives of this project are as follows:

1. Data Cleaning:

   - Remove duplicates and corrupted images from the dataset.
   - Address issues with an imbalanced dataset, where class distributions range from 5 to 60 samples per class.

2. Exploratory Data Analysis (EDA):

   - Perform EDA to understand the distribution of images, class imbalance, and gaps in the dataset.
   - Visualize and analyze the dataset's properties, such as image sizes and class distributions.

3. Data Splitting:

   - Simulate a scenario with few labeled images by splitting the dataset into labeled and unlabeled portions.
   - Use the unlabeled data for iterative labeling, starting with a small number of images per class.

4. Few-Shot Learning:

   - Build a classifier using few-shot learning techniques, leveraging the limited labeled data.
   - Gradually improve model accuracy by including more labeled data from the best predictions of the few-shot model.

5. Pre-trained Model with Fine-Tuning:

   - Build a classifier using a pre-trained architecture (e.g., ResNet18) and fine-tune the model for bird species classification.
   - Train the classifier head and fine-tune the backbone for better performance.

6. Model Evaluation:

   - Evaluate the performance of both the few-shot and pre-trained models on the combined test dataset.
   - Understand the strengths and weaknesses of both approaches.

7. LIME Interpretability:
   - Use LIME to interpret model predictions and understand which parts of the images are being used for classification.
   - Analyze the best and worst predictions, especially focusing on cases where the model does not use the bird itself for classification.

## Conclusion

Key conclusions from the project:

1. Model Performance:

   - The few-shot classifier achieved an accuracy of 73.92%.
   - The pre-trained model with fine-tuning achieved a test accuracy of 72.28%.
   - Both models performed similarly, but they were built using different techniques, making direct comparison less meaningful.

2. Insights from LIME:

   - The LIME interpretability technique revealed that in the case of the best predictions, the model used parts of the bird itself to make decisions. However, for the worst predictions, the model often used irrelevant areas outside the bird, leading to incorrect classifications.

3. Data Quality:

   - Data quality issues such as duplicates and imbalanced datasets were handled during the project, though some challenges remained, such as gaps in labels and variations in the number of samples per class.

4. Future Improvements:
   - Further improvements can be made by exploring advanced techniques like data augmentation, ensemble models, and more sophisticated transfer learning strategies. Additionally, improving the data split strategy and reducing class imbalances could further boost model performance.

## Technologies Used

- Python 3.x: The primary programming language used for model development.
- PyTorch: For deep learning model implementation.
- LIME: For model interpretability and visualizing the decision-making process.
- ResNet18: Pre-trained model used for transfer learning and fine-tuning.
- Matplotlib & Seaborn: For data visualization and plotting.
- scikit-learn: For evaluation metrics and confusion matrix.
- Google Colab: For cloud-based GPU training.

## How to Use

### Prerequisites

- Python 3.8 or later
- pip for managing Python packages

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <module-folder>
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
