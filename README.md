# DASC41103 Project 1 – Group 23

***

## 🚀 Overview
This project applies several machine learning algorithms—**Perceptron**, **Adaline**, **Logistic Regression**, and **SVM**—to predict whether an individual earns more or less than $50,000 per year using the UCI Adult Income dataset. The workflow includes:
- Data preprocessing
- Model training
- Evaluation
- Analysis of feature importance and decision boundaries

***

## 📁 Repository Structure

```text
notebooks/
│   ├── Adaline.ipynb                # Adaline algorithm implementation
│   ├── Perceptron.ipynb             # Perceptron model and analysis
│   ├── SVM.ipynb                    # SVM training and visualization
│   ├── logistic_regression.ipynb    # Logistic regression modeling
│   └── preprocessing.ipynb          # Data cleaning and feature scaling

data/
│   ├── project_adult.csv            # Main dataset
│   └── project_validation_inputs.csv# Validation set

predictions/
│   ├── Group_23_Adaline_PredictedOutputs.csv
│   ├── Group_23_LogisticRegression_PredictedOutputs.csv
│   ├── Group_23_Perceptron_PredictedOutputs.csv
│   └── Group_23_SVM_PredictedOutputs.csv

README.md                           # Project summary and analysis
```

***

## 📦 Folder Contents
- **notebooks/**: Jupyter notebooks for each algorithm and preprocessing steps.
- **data/**: Raw and validation datasets used for training and testing.
- **predictions/**: Model output files containing predicted labels for the validation set.
- **README.md**: This file, with project overview, structure, and analysis questions.

***

## 📝 Analytical Questions & Answers

### a. Why is feature scaling important for gradient-based algorithms?
Feature scaling ensures that all features contribute equally to the model's learning process. Without scaling, features with larger numeric ranges can dominate the gradient updates, causing inefficient, zig-zagging convergence and slow learning. Scaling (e.g., standardization) makes the cost function contours more circular, allowing gradient descent to converge faster and more reliably.

***

### b. Explain the difference between batch gradient descent and stochastic gradient descent.
- **Batch Gradient Descent** computes the gradient using the entire training dataset for each update. This leads to stable but potentially slow convergence, especially with large datasets.
- **Stochastic Gradient Descent (SGD)** updates the model parameters using only a single data point at a time. This results in faster, noisier updates that can help escape local minima and are more scalable for large datasets.

***

### c. Why does scikit-learn Perceptron and Adaline outperform book code?
Scikit-learn's implementations are highly optimized, using efficient numerical libraries (like BLAS) and providing features such as adaptive learning rates, regularization (L1/L2), and early stopping. These enhancements improve convergence speed and generalization. In contrast, book code often uses fixed learning rates, lacks regularization, and is not optimized for performance, making it less robust and more prone to overfitting or slow learning.

***

### d. Compare the decision boundaries of logistic regression and SVM.
- **Logistic Regression** produces linear decision boundaries, which are easy to interpret but may not capture complex relationships if the data is not linearly separable.
- **SVM** (with a linear kernel) also produces linear boundaries, but with nonlinear kernels, SVM can create flexible, nonlinear boundaries that better fit complex data patterns. SVMs also focus on maximizing the margin between classes, which can improve generalization.

***

### e. What is the role of regularization in preventing overfitting?
Regularization adds a penalty to the loss function for large model weights, discouraging overly complex models that fit the training data too closely. This helps the model generalize better to unseen data by controlling variance and reducing the risk of overfitting.

***

### f. Vary the C values of the scikit-learn LogisticRegression and linear SVC models with [0.01, 1.0, 100.0]. Discuss the impact.
- **Low C (0.01):** Strong regularization, leading to simpler models that may underfit the data.
- **Medium C (1.0):** Balanced regularization, often providing a good trade-off between bias and variance.
- **High C (100.0):** Weak regularization, allowing the model to fit the training data more closely, which can lead to overfitting.

As C increases, the model becomes more flexible but risks capturing noise in the training data. Validation accuracy and error analysis help determine the optimal C value for generalization.

***

## 👥 Authors
- Jordan Shortt
- Jake Laurie

***

For more details and visualizations, see the Jupyter notebooks and the project presentation.
