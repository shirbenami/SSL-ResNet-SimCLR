# SimCLR-Based Training and Fine-Tuning on STL10 Dataset

This repository contains a deep learning project focused on implementing SimCLR (Simple Contrastive Learning of Representations) for self-supervised learning and fine-tuning it on the STL10 dataset. The project includes training, validation, and testing modules, as well as visualization and evaluation tools.

## Project Overview

This project explores self-supervised learning techniques using SimCLR and evaluates its impact on classification performance when fine-tuned on a labeled dataset. The key steps include:

1.  **Supervised Training Baseline:**
   - Trained ResNet50 from scratch on the STL10 labeled dataset as a baseline for comparison.
   - 
2. **Self-Supervised Pretraining (SimCLR):**
   - Pretrained ResNet50 on the STL10 dataset's unlabeled images.
   - Optimized using the InfoNCE loss function to learn useful feature representations.
     ![ssl_losses_graphs](https://github.com/user-attachments/assets/7f38daa0-23cf-4bd0-8305-a303af3f6981)


3. **Fine-Tuning and Evaluation:**
   - Fine-tuned the pretrained ResNet50 on the STL10 dataset's labeled training set.
   - Evaluated on the validation and test sets.
   -  Compared the performance of the SSL-trained model with the supervised baseline.
   - Visualized performance metrics such as loss, accuracy, and confusion matrix.


   - 
3. **features_quality_check:**
  Before fine-tuning the SSL-pretrained ResNet50 on the labeled portion of the STL10 dataset, the feature representations learned by the model were evaluated using dimensionality reduction and clustering techniques. The goal was to assess the quality of the features extracted by the SimCLR pretraining process.

  1. **PCA (Principal Component Analysis)**:
   - PCA was applied to reduce the dimensionality of the feature space from 2048 to 2, allowing visualization of how well the model’s feature representations are clustered or spread out.
   - The results showed the dispersion of the features, indicating how well the model was able to differentiate between different classes based on the learned features.
   - The dispersion of the points suggests that there isn't a clear separation between the categories, meaning that the model hasn’t yet learned features that distinctly separate different classes.
   - Ideally, if the features were more discriminative, we would expect to see the points forming distinct clusters corresponding to different classes.

   ![pca_plot](https://github.com/user-attachments/assets/faf7f286-734a-44cf-a6d6-0538d12ab0ff)

  2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
   - t-SNE was used for further visualization in a 2D space, helping to better understand the similarity relationships between data points.
   - t-SNE attempts to maintain local relationships and group similar images together in the lower-dimensional space, providing insights into how effectively the model learned class-related features.
   - In this graph, we see the feature representations plotted in a 2D space. The feature extractor's output is spread across the plot in a cloud-like shape. This indicates that the features are not clearly clustered or separated by class. The dispersion suggests that the model hasn't yet learned to distinctly separate different categories or features in the lower-dimensional space.

![tsne_plot](https://github.com/user-attachments/assets/c81c162b-f75a-466a-866d-99df2b5c736a)

  3. **k-Means Clustering**:
   - k-Means clustering was applied to group the extracted features into clusters. By comparing the clusters with the actual labels, the model's ability to separate different classes was assessed.
   - The clustering results provided a way to evaluate how well the self-supervised learning method managed to capture the underlying structure of the data without using labeled examples.
   - In this,the different colors represent different clusters formed by the k-Means algorithm. These clusters suggest that, although the features may not have been clearly separated initially (as shown in the first graph), the model has managed to group similar data points together based on the feature representations. The separation into distinct clusters indicates that the model was able to capture some structure of the data even without labels.

![kmeans_plot](https://github.com/user-attachments/assets/dd6ed64a-02d6-490b-999d-fc3b272a9603)

  **What We Learn**:
  - **Clustering Success**: The k-Means clustering graph shows that distinct groups or clusters have been formed, which suggests that the feature extractor has captured meaningful patterns from the data. The model is able to differentiate between different categories, even though this wasn't as clear in the t-SNE visualization.
  - **Significance of Clustering**: The fact that k-Means can form clusters despite the first t-SNE graph's dispersed pattern indicates that the self-supervised learning model is indeed learning useful feature representations. These clusters could be aligned with the actual classes in the dataset, providing valuable insights into how the model is performing before fine-tuning.


  These techniques were employed to gain a deeper understanding of the effectiveness of the feature extractor before applying the fine-tuning process on the labeled dataset.



## Dataset - STL10

The STL10 dataset is designed for developing self-supervised learning techniques. It includes:
- **Unlabeled Set:** 100,000 images (for SSL training).
- **Train Set:** 5,000 labeled images across 10 classes (500 per class).
- **Test Set:** 8,000 labeled images across the same 10 classes.
- The dataset is loaded using PyTorch's `torchvision.datasets.STL10` utility.

## Project Structure

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







