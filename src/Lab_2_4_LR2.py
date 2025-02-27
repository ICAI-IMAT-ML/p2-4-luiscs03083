import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model


        # Calcular los coeficientes usando la ecuación normal
        beta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Separar intercepto y coeficientes
        self.intercept = beta[0]  # Primer valor es el intercepto
        self.coefficients = beta[1:]  # Resto de valores son los coeficientes

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)

        # Inicializar coeficientes y el intercepto en valores pequeños
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01  # Pequeños valores aleatorios
        self.intercept = np.random.rand() * 0.01  # Pequeño valor aleatorio

        # Implement gradient descent ()
        for epoch in range(iterations):
            predictions = X[:,1:] @ self.coefficients + self.intercept 
            error = predictions - y

            # Write the gradient values and the updates for the paramenters
            gradient_w = (1 / m) * (X[:,1:].T @ error)

            gradient_b = (1 / m) * np.sum(error)

            # Actualizar parámetros con descenso de gradiente
            self.coefficients -= learning_rate * gradient_w
            self.intercept -= learning_rate * gradient_b

            # Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.mean(error**2)

                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        
        if X.ndim == 1:
           X = X.reshape(-1, 1)

        
        return X @ self.coefficients + self.intercept


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Suma total de cuadrados
    ss_residual = np.sum((y_true - y_pred) ** 2)  # Suma de los errores al cuadrado
    if ss_total == 0:
        r_squared = np.nan  # O establecerlo en 0
    else:
        r_squared = 1 - (ss_residual / ss_total)

    # Root Mean Squared Error
    
    rmse = np.sqrt(np.sum((y_true-y_pred)**2)/len(y_pred))

    # Mean Absolute Error
    
    mae = np.sum(abs(y_true-y_pred))/len(y_pred)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()

    for index in sorted(categorical_indices, reverse=True):  
        # Extraer la columna categórica
        categorical_column = X_transformed[:, index]

        # Encontrar valores únicos en la columna
        unique_values = np.unique(categorical_column)

        # Crear la matriz one-hot
        one_hot = np.zeros((X_transformed.shape[0], len(unique_values)))
        for row_idx, value in enumerate(categorical_column):
            category_idx = np.where(unique_values == value)[0][0]  # Encuentra el índice de la categoría
            one_hot[row_idx, category_idx] = 1  # Asigna un único 1

        # Si drop_first=True, eliminamos la primera columna para evitar multicolinealidad
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Eliminar la columna original e insertar nuevas columnas one-hot en la misma posición
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))

    return X_transformed.astype(float)  # Convertimos a float para ML
