import catboost as cb
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split

from optcat.core import CatBoostClassifier, CatBoostRegressor

N_TRIALS = 2
ITERATIONS = 10
RANDOM_STATE = 41
SEED = 42


def test_predict():
    params = {
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "iterations": 1,
    }
    model = CatBoostClassifier(params=params, n_trials=1)
    data, target = datasets.load_breast_cancer(return_X_y=True)
    model.fit(X=data, y=target)
    y_pred = model.predict(data)
    assert target.shape == y_pred.shape


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


def test_score():
    params = {
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "iterations": ITERATIONS,
    }
    data, target = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, random_state=RANDOM_STATE
    )

    def get_score(_model):
        _model.fit(X=x_train, y=y_train)
        return _model.score(x_test, y_test)

    origin_score = get_score(cb.CatBoostClassifier(**params))
    tuner_score = get_score(CatBoostClassifier(params))

    assert origin_score <= tuner_score
