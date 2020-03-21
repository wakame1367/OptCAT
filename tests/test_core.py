import pytest
from sklearn import datasets

from optcat.core import CatBoostClassifier, CatBoostRegressor

N_TRIALS = 2
ITERATIONS = 10


def test_classifier_fit():
    params = {
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "iterations": ITERATIONS,
    }
    model = CatBoostClassifier(params=params, n_trials=N_TRIALS)
    data, target = datasets.load_breast_cancer(return_X_y=True)
    model.fit(X=data, y=target)


@pytest.mark.parametrize("metric_name", ["loss_function", "objective"])
def test_fit_with_param_alias(metric_name):
    params = {
        "bootstrap_type": "Bayesian",
        metric_name: "Logloss",
        "iterations": ITERATIONS,
    }
    model = CatBoostClassifier(params=params, n_trials=N_TRIALS)
    data, target = datasets.load_breast_cancer(return_X_y=True)
    model.fit(X=data, y=target)

    params = {
        "bootstrap_type": "Bayesian",
        metric_name: "RMSE",
        "iterations": ITERATIONS,
    }
    model = CatBoostRegressor(params=params, n_trials=N_TRIALS)
    data, target = datasets.load_boston(return_X_y=True)
    model.fit(X=data, y=target)


def test_regressor_fit():
    params = {
        "bootstrap_type": "Bayesian",
        "loss_function": "RMSE",
        "iterations": ITERATIONS,
    }
    model = CatBoostRegressor(params=params, n_trials=N_TRIALS)
    data, target = datasets.load_boston(return_X_y=True)
    model.fit(X=data, y=target)
