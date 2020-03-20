# OptCAT

<p align="center">
<a href="https://github.com/wakamezake/OptCAT/actions"><img alt="Actions Status" src="https://github.com/wakamezake/OptCAT/workflows/Python package/badge.svg"></a>
<a href="https://pypi.org/project/optcat/"><img alt="Actions Status" src="https://badge.fury.io/py/optcat.svg"></a>
<a href="https://github.com/wakamezake/OptCAT/master/LICENSE"><img alt="License: MIT" src="http://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

OptCAT (= [Optuna][1] + [CatBoost][2]) provides a scikit-learn compatible estimator that tunes hyperparameters in CatBoost with Optuna.

This Repository is very influenced by [Y-oHr-N/OptGBM](https://github.com/Y-oHr-N/OptGBM).

## Examples

```python:classification.py
from optcat.core import CatBoostClassifier
from sklearn import datasets

params = {
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "iterations": 100
    }

model = CatBoostClassifier(params=params, n_trials=5)
data, target = datasets.load_breast_cancer(return_X_y=True)
model.fit(X=data, y=target)
```

## Installation

```
pip install optcat
```

## Testing

```
poetry run pytest
```


[1]: https://optuna.org/
[2]: https://catboost.ai/
