import copy
from typing import Dict, Any, List, Callable, Optional, Union, Iterable

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
        callbacks: Optional[List[Callable]] = None,
        categorical_feature: Union[List[int], List[str], str] = "auto",
        cv: Optional[CVType] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        feature_name: Union[List[str], str] = "auto",
        f_eval: Optional[Callable] = None,
        f_obj: Optional[Callable] = None,
        n_estimators: int = 100,
        param_distributions: Optional[
            Dict[str, distributions.BaseDistribution]
        ] = None,
    ) -> None:
        self.callbacks = callbacks
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.eval_name = eval_name
        self.feature_name = feature_name
        self.feval = f_eval
        self.fobj = f_obj
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
            num_boost_round=self.n_estimators,
        )  # Dict[str, List[float]]
        value = eval_hist["{}-mean".format(self.eval_name)][-1]  # type: float

        return value

    def _get_params(self, trial: trial_module.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["feature_fraction"] = trial.suggest_discrete_uniform(
                "feature_fraction", 0.1, 1.0, 0.05
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
            params["lambda_l1"] = trial.suggest_loguniform(
                "lambda_l1", 1e-09, 10.0
            )
            params["lambda_l2"] = trial.suggest_loguniform(
                "lambda_l2", 1e-09, 10.0
            )

            if params["boosting_type"] != "goss":
                params["bagging_fraction"] = trial.suggest_discrete_uniform(
                    "bagging_fraction", 0.5, 0.95, 0.05
                )
                params["bagging_freq"] = trial.suggest_int(
                    "bagging_freq", 1, 10
                )

            return params

        for name, distribution in self.param_distributions.items():
            params[name] = trial._suggest(name, distribution)

        return params
