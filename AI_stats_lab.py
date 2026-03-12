"""
AIstats_lab.py
Student starter file for the Regularization & Overfitting lab.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# =========================
# Helper Functions
# =========================
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# =========================
# Q1 Lasso Regression
# =========================
def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """
    # TODO: Load diabetes dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # TODO: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TODO: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TODO: Add bias column
    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)

    # TODO: Initialize theta
    n_features = X_train_b.shape[1]
    theta = np.zeros(n_features)

    # TODO: Implement gradient descent with L1 regularization
    n = len(y_train)
    for _ in range(epochs):
        y_pred = X_train_b @ theta
        error = y_pred - y_train

        grad = (1 / n) * (X_train_b.T @ error)
        l1_grad = np.sign(theta)
        l1_grad[0] = 0

        grad += lambda_reg * l1_grad
        theta -= lr * grad

    # TODO: Compute predictions
    train_preds = X_train_b @ theta
    test_preds = X_test_b @ theta

    # TODO: Compute metrics
    train_mse = mse(y_train, train_preds)
    test_mse = mse(y_test, test_preds)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    return train_mse, test_mse, train_r2, test_r2, theta

# =========================
# Q2 Polynomial Overfitting
# =========================
def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """
    # TODO: Load dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # TODO: Select BMI feature only
    X_bmi = X[:, 2].reshape(-1, 1)

    # TODO: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bmi, y, test_size=0.2, random_state=42
    )

    degrees = []
    train_errors = []
    test_errors = []

    # TODO: Loop through polynomial degrees
    for degree in range(1, max_degree + 1):
        # TODO: Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # TODO: Fit regression using normal equation
        X_train_b = add_bias(X_train_poly)
        X_test_b = add_bias(X_test_poly)

        try:
            theta = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
        except np.linalg.LinAlgError:
            continue

        # TODO: Compute train/test errors
        train_pred = X_train_b @ theta
        test_pred = X_test_b @ theta

        degrees.append(degree)
        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test, test_pred))

    return {
        "degrees": degrees,
        "train_mse": train_errors,
        "test_mse": test_errors,
    }
