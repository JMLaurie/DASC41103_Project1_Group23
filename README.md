# DASC41103_Project1_Group23
Project Overview
This project uses machine learning algorithms (Perceptron, Logistic Regression, Adaline, SVM) to predict whether a person makes more or less than $50,000/year using the UCI Adult Income dataset. The workflow includes preprocessing, hyperparameter tuning, model evaluation, and feature importance analysis.

Repository Structure
notebooks/

Adaline.ipynb: Implementation and analysis of Adaline algorithm

Perceptron.ipynb: Perceptron model building and performance plots

SVM.ipynb: SVM training, validation, and decision boundary visualization

logistic_regression.ipynb: Logistic regression modeling and feature importance

preprocessng.ipynb: Data cleaning, encoding, and scaling routines

data/

project_adult.csv: Main dataset for model training and testing

project_validation_inputs.csv: Validation set for final predictions

predictions/

Group_23_Adaline_PredictedOutputs.csv: Adaline output predictions

Group_23_LogisticRegression_PredictedOutputs.csv: Logistic Regression output predictions

Group_23_Perceptron_PredictedOutputs.csv: Perceptron predictions

Group_23_SVM_PredictedOutputs.csv: SVM model predictions

README.md
Project summary, folder structure, and answers to analysis questions (see below).

Analytical Questions & Answers
a. Why is feature scaling important for gradient-based algorithms?
Feature scaling ensures all features contribute equally to gradient updates. Without scaling, features with larger numeric ranges dominate, causing slow zig-zag convergence and inefficiency. Scaling produces more 'circular' cost contours, allowing gradient descent to converge faster and more reliably.

b. Explain the difference between batch gradient descent and stochastic gradient descent.
Batch Gradient Descent: Uses the entire training dataset for each update, leading to stable but slow convergence.

Stochastic Gradient Descent: Updates weights using only a single data point at a time, offering faster and noisier convergence. Stochastic methods are better for large datasets and escaping local minima.

c. Why does scikit-learn Perceptron and Adaline outperform book code?
Scikit-learn implementations leverage optimized backends (like BLAS, C/Fortran), have adaptive learning rates, built-in regularization (L1/L2), and early stopping. Book code is less efficient, uses fixed learning rates, lacks regularization, and typically stops after a fixed number of epochs, making it less robust and prone to overfitting or slow convergence.

d. Compare the decision boundaries of logistic regression and SVM.
Logistic Regression produces linear decision boundaries, leading to easier interpretation but potential limitations if the relationship is nonlinear.

SVM (especially with nonlinear kernels) creates flexible, nonlinear boundaries, better fitting complex patterns but possibly leading to overfitting if not regularized properly.

e. What is the role of regularization in preventing overfitting?
Regularization adds a complexity penalty to the model’s loss function, keeping weights small and limiting model complexity. Lower regularization ('C' parameter) makes the model simpler and less likely to overfit, while higher values make it more flexible, increasing overfitting risk if not tuned carefully.

f. Varying the C values ([0.01, 1.0, 100.0]) in scikit-learn Logistic Regression and linear SVC models
Low C (0.01): Strong regularization, prevents overfitting but may underfit, lowering accuracy.

High C (100.0): Weak regularization, fits training data closely and may overfit.

C of 1.0: Balanced default, good trade-off between bias and variance. Validation accuracy and error curves help empirically find best C for generalization.

Authors
Jordan Shortt
Jake Laurie

For more details and visuals, see the attached presentation and Jupyter notebooks.
