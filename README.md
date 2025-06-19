# Multi-label Image Classification using CNN

This project aims to classify images into multiple labels based on the given dataset. The goal is to predict the correct labels associated with each image, where each image can have multiple labels simultaneously.

## Dataset

The dataset used in this project is the [Multi-label Image Classification Dataset](https://www.kaggle.com/datasets/meherunnesashraboni/multi-label-image-classification-dataset/data) from Kaggle. This dataset contains images categorized under multiple labels, making it suitable for multi-label classification tasks.

## Objective

The objective of this project is to predict the correct labels for each image using a Convolutional Neural Network (CNN). The CNN will learn to recognize patterns in the images and classify them into one or more categories.

## Requirements

- Python 3.x
- TensorFlow (for CNN model)
- scikit-multilearn (for multi-label classification)
- scikit-learn (for train-test splitting)
- tqdm (for progress bar)
- Matplotlib and Seaborn (for data visualization)
- PIL (Python Imaging Library)

## Libraries Used

- TensorFlow - For building and training the CNN model.
- scikit-learn - For data preprocessing and splitting.
- scikit-multilearn - For handling multi-label classification.
- Matplotlib/Seaborn - For data visualization.
- PIL - For image handling.

## Final Analysis

To evaluate the effectiveness of different modeling approaches for multi-label image classification, 3 models were compared:
* Baseline CNN
* Tuned CNN
* Transfer Learning (MobileNetV2)

The following table summarizes the performance across key metrics:

| **Metric**                    | **Baseline CNN**  | **Tuned CNN**              | **Transfer Learning (MobileNetV2)** |
| ----------------------------- | ----------------- | -------------------------- | ----------------------------------- |
| **Sample-based F1-score**     | 0.64              | 0.72                       | **0.97**                            |
| **Subset Accuracy**           | 0.62 | 0.685                      | ~ 0.90 **(estimated)**               |
| **Avg. Per-label Accuracy**   | 0.954     | 0.958                      | **0.97**                            |
| **Validation Accuracy (max)** | \~0.64 (inferred) | 0.80                       | **0.922**                           |
| **Strongest Labels**          | `tabla`, `cycle`  | `tabla`, `sitar`, `flutes` | `tabla`, `flutes`, `harmonium`      |
| **Weakest Labels**            | `truck`, `bus`    | `bus`                      | *None significantly weak*           |
| **Trainable Params**          | \~22M             | \~22M (tuned)              | **\~330K (fine-tuned)**                 |

**The baseline CNN** provided a starting point with modest performance, achieving a subset accuracy of 0.6193 and a sample-based F1-score of 0.64, with strong per-label accuracy (average: 0.9537) but low recall on certain classes like `truck` and `bus`.

After **hyperparameter tuning**, the CNN showed a notable improvement, with a higher F1-score (0.72) and subset accuracy (0.685), reflecting better balance and more stable predictions across labels after optimizing filter sizes, dense units, dropout, and learning rate.

However, **the transfer learning model using MobileNetV2** dramatically outperformed both CNN variants. It achieved a near-perfect sample-based F1-score of 0.97, significantly higher than the tuned CNN’s 0.72. With an estimated subset accuracy of ~0.90, and consistentyly high per-label performance, it demonstrated strong generalization across all 10 target labels. Even previously weak classes like truck and bus showed substantial gains.

Importantly, it did so with only ~330K trainable parameters (compared to ~22 million in the custom CNN), showing that leveraging pretrained knowledge not only boosts accuracy but also reduces training cost and overfitting risk.

**Conclusion**  
The results clearly demonstrate the superiority of transfer learning for multi-label classification tasks with limited data. While baseline and tuned CNNs showed progressive improvement, the MobileNetV2 model with fine-tuning offered the best trade-off between performance and computational efficiency. For future work or deployment, transfer learning should be considered the default approach, especially when pretrained models on large-scale datasets like ImageNet are available.
