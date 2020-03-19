import copy
from typing import Dict, Any, Callable, Optional, Union, Iterable

import catboost as cb
from optuna import distributions
from optuna import trial as trial_module
from sklearn.model_selection import BaseCrossValidator

CVType = Union[int, Iterable, BaseCrossValidator]


class _Objective:
    def __init__(
        self,
        params: Dict[str, Any],
        dataset: cb.Pool,
        eval_name: str,
        is_higher_better: bool,
        n_samples: int,
        cv: Optional[CVType] = None,
        early_stopping_rounds: Optional[int] = None,
        n_estimators: int = 100,
        param_distributions: Optional[
            Dict[str, distributions.BaseDistribution]
        ] = None,
    ) -> None:
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_name = eval_name
        self.is_higher_better = is_higher_better
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.params = params
        self.param_distributions = param_distributions

    def __call__(self, trial: trial_module.Trial) -> float:
        params = self._get_params(trial)  # type: Dict[str, Any]
        dataset = copy.copy(self.dataset)
        eval_hist = cb.cv(
            params=params,
            dtrain=dataset,
            early_stopping_rounds=self.early_stopping_rounds,
            folds=self.cv,
            iterations=self.n_estimators,
            as_pandas=False
        )  # Dict[str, List[float]]
        value = eval_hist["test-{}-mean".format(self.eval_name)][-1] # type: float
        return value

    def _get_params(self, trial: trial_module.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["colsample_bylevel"] = trial.suggest_discrete_uniform(
                "colsample_bylevel", 0.1, 1.0, 0.05
            )
            params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            params["num_leaves"] = trial.suggest_int(
                "num_leaves", 2, 2 ** params["max_depth"]
            )
            # See https://github.com/Microsoft/LightGBM/issues/907
            params["min_data_in_leaf"] = trial.suggest_int(
                "min_data_in_leaf",
                1,
                max(1, int(self.n_samples / params["num_leaves"])),
            )
            params["l2_leaf_reg"] = trial.suggest_loguniform(
                "lambda_l2", 1e-09, 10.0
            )

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_discrete_uniform(
                    "bagging_temperature", 0.5, 0.95, 0.05
                )
            elif params["bootstrap_type"] == "Bernoulli" or \
                params["bootstrap_type"] == "Poisson":
                params["subsample"] = trial.suggest_uniform(
                    "subsample", 0.1, 1
                )

            return params

        for name, distribution in self.param_distributions.items():
            params[name] = trial._suggest(name, distribution)

        return params
