# SimCLR-Based Training and Fine-Tuning on STL10 Dataset
![simclr_contrastive_learning](https://github.com/user-attachments/assets/7488a8e1-e232-4dc4-8405-082f2128c59d)

This repository contains a deep learning project focused on implementing SimCLR (Simple Contrastive Learning of Representations) for self-supervised learning and fine-tuning it on the STL10 dataset. The project includes training, validation, and testing modules, as well as visualization and evaluation tools.

Resnet50 Architecture:
![resnet_bannner (1)](https://github.com/user-attachments/assets/f46486ca-ff3e-4499-a6d5-e3b2159d7ec6)


## Project Overview

This project explores self-supervised learning techniques using SimCLR and evaluates its impact on classification performance when fine-tuned on a labeled dataset. The key steps include:

1.  **Supervised Training Baseline:**
   - Trained ResNet50 from scratch on the STL10 labeled dataset as a baseline for comparison.
     
2. **Self-Supervised Pretraining (SimCLR):**
   - Pretrained ResNet50 on the STL10 dataset's unlabeled images.
   - Optimized using the InfoNCE loss function to learn useful feature representations.

3. **Fine-Tuning and Evaluation:**
   - Fine-tuned the pretrained ResNet50 on the STL10 dataset's labeled training set.
   - Evaluated on the validation and test sets.
   -  Compared the performance of the SSL-trained model with the supervised baseline.
   - Visualized performance metrics such as loss, accuracy, and confusion matrix.

4. **features_quality_check:**
  Before fine-tuning the SSL-pretrained ResNet50 on the labeled portion of the STL10 dataset, the feature representations learned by the model were evaluated using dimensionality reduction and clustering techniques. The goal was to assess the quality of the features extracted by the SimCLR pretraining process.

   4.1. **PCA (Principal Component Analysis)**:
      - PCA was applied to reduce the dimensionality of the feature space from 2048 to 2, allowing visualization of how well the model’s feature representations are clustered or spread out.
      - The results showed the dispersion of the features, indicating how well the model was able to differentiate between different classes based on the learned features.

   4.2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
      - t-SNE was used for further visualization in a 2D space, helping to better understand the similarity relationships between data points.
      - t-SNE attempts to maintain local relationships and group similar images together in the lower-dimensional space, providing insights into how effectively the model learned class-related features.


   4.3. **k-Means Clustering**:
      - k-Means clustering was applied to group the extracted features into clusters. By comparing the clusters with the actual labels, the model's ability to separate different classes was assessed.
      - The clustering results provided a way to evaluate how well the self-supervised learning method managed to capture the underlying structure of the data without using labeled examples.


## Dataset - STL10

The STL10 dataset is designed for developing self-supervised learning techniques. It includes:
- **Unlabeled Set:** 100,000 images (for SSL training).
- **Train Set:** 5,000 labeled images across 10 classes (500 per class).
- **Test Set:** 8,000 labeled images across the same 10 classes.
- The dataset is loaded using PyTorch's `torchvision.datasets.STL10` utility.

## Project Structure
```python

project_root/
├── dataset/
│   └── stl10_loader.py          # Responsible for loading and preprocessing the STL10 dataset.
│
├── loss_functions/
│   └── info_nce.py              # Implementation of the InfoNCE loss function used in SimCLR.
│
├── model/
│   └── resnet50.py              # Defines the ResNet50 architecture with optional modifications (e.g., fine-tuning or custom classification head).
│
├── trainers/
│   ├── train.py                 # Implements the training loop for supervised and SSL models.
│   ├── validate.py              # Implements the validation loop for calculating validation loss and accuracy.
│   └── test.py                  # Implements the testing loop for evaluating the model on the test dataset.
│
├── output/
│   ├── logs/                    # Stores loss and accuracy graphs, as well as confusion matrix images.
│   └── models/                  # Stores the trained model weights (.pth files) for SimCLR and fine-tuning.
│
├── simclr_train.py              # Main script for training SimCLR on the STL10 unlabeled dataset.
├── supervised_train.py          # Main script for supervised training on the STL10 labeled dataset.
├── fine_tuning.py               # Main script for fine-tuning a model using SSL-pretrained weights.
├── features_quality_check.py    # Main script for features_quality_check a model using SSL-pretrained weights.
├── README.md                    # Project description, instructions, and results.
└── .gitignore                   # Specifies files and folders to exclude from version control.
```




## Evaluation Results and Insights

### Supervised_training VS Fine Tune:

#### **Confusion Matrix for Fine-Tuned Model (60 Epochs)**
![confusion_matrix_fine_tuning_60epochs](https://github.com/user-attachments/assets/fd22537f-3686-46c7-bde6-f785e9d7e5a7)

The confusion matrix displayed shows the results of the model's performance after fine-tuning for 60 epochs. The model's predictions are compared with the true labels across all 10 classes of the STL10 dataset.

- The **diagonal elements** represent the correct classifications, with high values on the diagonal indicating strong performance in predicting the correct class for most images.
- The **off-diagonal elements** show misclassifications, with smaller values suggesting where the model may have struggled.
- In the fine-tuned model's confusion matrix, the model achieves high accuracy in predicting most classes, though there are some misclassifications particularly for certain classes (e.g., class 3 and class 8).


#### **Confusion Matrix for Initial Supervised Training (60 Epochs)**
![confusion_matrix_60epochs](https://github.com/user-attachments/assets/0e25c670-1bae-49e0-8b49-76c7ffc041ad)

The confusion matrix for the supervised model trained on the STL10 labeled dataset for 60 epochs shows the initial performance of the model before fine-tuning.

- Similar to the fine-tuned model's confusion matrix, the **diagonal elements** show the correct predictions, with some misclassifications visible off the diagonal.
- Compared to the fine-tuned model, there are more misclassifications across all classes, indicating that the fine-tuned model has improved performance by learning better feature representations.

#### **Loss and Accuracy Curves for Fine-Tuned Model (60 Epochs)**
![fine_tuning_classification_graphs_60epochs](https://github.com/user-attachments/assets/1d5ac73a-88d0-40ef-913e-e75ef37d76b1)

These graphs show the training loss and accuracy over the course of 60 epochs for the fine-tuned model.

- **Loss Graph**: The **training loss** steadily decreases as the model learns, with some fluctuations in the validation loss. However, both the training and validation losses reach relatively stable values by the end of the training.
- **Accuracy Graph**: The **train accuracy** steadily increases, while the **validation accuracy** fluctuates initially but increases overall. After 60 epochs, the validation accuracy reaches around 84%, indicating that the model successfully learned to generalize.

#### **Loss and Accuracy Curves for Supervised Model (60 Epochs)**
![train_val_graphs_supervised_model_60epochs](https://github.com/user-attachments/assets/4d8175ea-e5c7-4e0e-9696-c3e05132f497)

These graphs represent the training loss and accuracy over 60 epochs for the supervised model trained on the labeled dataset without any self-supervised pretraining.

- **Loss Graph**: The **training loss** decreases more slowly than in the fine-tuned model, with more noticeable fluctuations in the validation loss.
- **Accuracy Graph**: The **train accuracy** improves, but the **validation accuracy** increases more gradually and remains lower than the fine-tuned model, suggesting that the supervised model had a harder time generalizing to the validation set.

#### **Key Takeaways:**
1. The **fine-tuned model** shows a significant improvement over the **supervised baseline model**, achieving higher accuracy and a more stable performance in both training and validation.
2. The **confusion matrices** indicate that the fine-tuned model performs better in correctly classifying images across the various classes compared to the initial supervised model.
3. The **loss and accuracy graphs** show a clear improvement in performance for the fine-tuned model, especially in terms of generalization, as indicated by the higher validation accuracy.
4. Overall, the fine-tuning process using self-supervised learning with SimCLR has a positive impact on the model's ability to classify images more accurately.

     
### features_quality_check:

  Before fine-tuning the SSL-pretrained ResNet50 on the labeled portion of the STL10 dataset, the feature representations learned by the model were evaluated using dimensionality reduction and clustering techniques. The goal was to assess the quality of the features extracted by the SimCLR pretraining process.

  1. **PCA (Principal Component Analysis)**:
   ![pca_plot](https://github.com/user-attachments/assets/faf7f286-734a-44cf-a6d6-0538d12ab0ff)
   - The dispersion of the points suggests that there isn't a clear separation between the categories, meaning that the model hasn’t yet learned features that distinctly separate different classes.
   - Ideally, if the features were more discriminative, we would expect to see the points forming distinct clusters corresponding to different classes.


  2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
   ![tsne_plot](https://github.com/user-attachments/assets/c81c162b-f75a-466a-866d-99df2b5c736a)

   - In this graph, we see the feature representations plotted in a 2D space. The feature extractor's output is spread across the plot in a cloud-like shape. This indicates that the features are not clearly clustered or separated by class. The dispersion suggests that the model hasn't yet learned to distinctly separate different categories or features in the lower-dimensional space.


  3. **k-Means Clustering**:
   ![kmeans_plot](https://github.com/user-attachments/assets/dd6ed64a-02d6-490b-999d-fc3b272a9603)

   - In this,the different colors represent different clusters formed by the k-Means algorithm. These clusters suggest that, although the features may not have been clearly separated initially (as shown in the first graph), the model has managed to group similar data points together based on the feature representations. The separation into distinct clusters indicates that the model was able to capture some structure of the data even without labels.
    

  **What We Learn**:
  - **Clustering Success**: The k-Means clustering graph shows that distinct groups or clusters have been formed, which suggests that the feature extractor has captured meaningful patterns from the data. The model is able to differentiate between different categories, even though this wasn't as clear in the t-SNE visualization.
  - **Significance of Clustering**: The fact that k-Means can form clusters despite the first t-SNE graph's dispersed pattern indicates that the self-supervised learning model is indeed learning useful feature representations. These clusters could be aligned with the actual classes in the dataset, providing valuable insights into how the model is performing before fine-tuning.


  These techniques were employed to gain a deeper understanding of the effectiveness of the feature extractor before applying the fine-tuning process on the labeled dataset.



## Resources
- [SimCLR Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [STL10 Dataset](https://cs.stanford.edu/~acoates/stl10/)






