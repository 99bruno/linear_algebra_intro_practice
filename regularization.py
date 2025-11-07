import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    return [X_train, X_test, y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_diabetes()
    X, y = data.data, data.target
    return preprocess(X, y)


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    return preprocess(X, y)


def _ensure_1d_y(y: np.ndarray) -> np.ndarray:
    """Flatten y to 1D for sklearn estimators."""
    return np.asarray(y).ravel()


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    y = _ensure_1d_y(y)
    model = LinearRegression()
    model.fit(X, y)
    return model

def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    y = _ensure_1d_y(y)
    ridge = Ridge(random_state=0)
    param_grid = {
        "alpha": np.logspace(-4, 3, 20),  # 1e-4 ... 1e3
    }
    gscv = GridSearchCV(
        ridge,
        param_grid=param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gscv.fit(X, y)
    return gscv.best_estimator_

def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    y = _ensure_1d_y(y)
    lasso = Lasso(random_state=0, max_iter=10000)
    param_grid = {
        "alpha": np.logspace(-4, 1, 20),  # 1e-4 ... 10
    }
    gscv = GridSearchCV(
        lasso,
        param_grid=param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gscv.fit(X, y)
    return gscv.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    y = _ensure_1d_y(y)
    clf = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(X, y)
    return clf


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    y = _ensure_1d_y(y)
    base = LogisticRegression(
        penalty="l2",
        solver="lbfgs",  # supports l2
        max_iter=1000,
        n_jobs=-1,
        random_state=0,
    )
    param_grid = {
        "C": np.logspace(-4, 3, 12),  # 1e-4 ... 1e3
    }
    gscv = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gscv.fit(X, y)
    return gscv.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    y = _ensure_1d_y(y)
    base = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=2000,  # saga can need more iterations
        n_jobs=-1,
        random_state=0,
    )
    param_grid = {
        "C": np.logspace(-4, 2, 12),  # 1e-4 ... 1e2
    }
    gscv = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gscv.fit(X, y)
    return gscv.best_estimator_