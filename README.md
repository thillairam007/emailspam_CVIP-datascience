# Spam Detection Project

## Overview

This project focuses on building a machine learning model for spam email detection using a Random Forest Classifier. The model is trained on a dataset consisting of email text and corresponding labels to classify emails as spam or not spam.

## Dataset

Dataset - [Spam.csv](https://drive.google.com/file/d/1IjahqEUflUKpPVA-mLt6uDR9-TcsSXa3/view?usp=sharing)

The dataset contains the following columns:
- `EmailText`: The text content of the email.
- `Label`: The label indicating whether the email is spam or not spam.

## Installation

1. Clone the Git repository:
```bash
git clone https://github.com/thillairam007/emailspam_CVIP-datascience
```
2. Install the required Python libraries (check requirement.txt)
3. Run the Jupyter Notebook spamclassification.ipynb

## Random Forest Model Development

In this section, we will outline the steps to develop a Random Forest model for spam email detection.

### Steps

1. **Data Preprocessing:**

   - Load the dataset.
   - Explore the dataset and handle missing data.
   - Preprocess the email text data (e.g., tokenization, TF-IDF vectorization).
   - Split the dataset into training and testing sets.

2. **Model Initialization:**

   - Import the Random Forest Classifier from scikit-learn.
   - Initialize the Random Forest Classifier with appropriate hyperparameters.

3. **Model Training:**

   - Train the Random Forest Classifier on the training data.

4. **Model Evaluation:**

   - Evaluate the model's performance on the testing data.
   - Calculate accuracy and generate a classification report.

5. **Test Case Validation:**

   - Validate the model using sample email texts.
   - Print the classification results for each sample email.

### Example Code

Here's an example code snippet to initialize and train the Random Forest Classifier:

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train_vec, y_train)
```

### License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/thillairam007/emailspam_CVIP-datascience/blob/main/LICENSE) file for details.