import logging
from typing import Dict, Any, Optional, Union

import catboost as cb
from optuna import distributions
from optuna import samplers
from optuna import study as study_module
from optuna import trial as trial_module
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.model_selection import check_cv

from .typing import (
    CVType,
    MultipleDataType,
    TargetDataType,
    PairsType,
    SampleWeightType,
    FeatureType,
)


# https://catboost.ai/docs/references/eval-metric__supported-metrics.html
def _is_higher_better(metric: str) -> bool:
    """

    Args:
        metric:

    Returns:

    >>> _is_higher_better("AUC")
    True
    >>> _is_higher_better("auc")
    False
    """
    higher_better_metrics = {
        "Recall",
        "Precision",
        "F1",
        "TotalF1",
        "Accuracy",
        "AUC",
        "R2",
        "BrierScore",
        "Kappa",
        "WKappa",
        "DCG",
        "NDCG",
    }
    return metric in higher_better_metrics


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
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
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
        eval_hist = cb.cv(
            params=params,
            dtrain=self.dataset,
            early_stopping_rounds=self.early_stopping_rounds,
            folds=self.cv,
            verbose=False,
            iterations=self.n_estimators,
            as_pandas=False,
        )  # Dict[str, List[float]]
        value = eval_hist["test-{}-mean".format(self.eval_name)][-1]  # type: float

        return value

    def _get_params(self, trial: trial_module.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["colsample_bylevel"] = trial.suggest_discrete_uniform(
                "colsample_bylevel", 0.1, 1.0, 0.05
            )
            params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            # https://catboost.ai/docs/concepts/parameter-tuning.html#tree-growing-policy
            # params["num_leaves"] = trial.suggest_int(
            #     "num_leaves", 2, 2 ** params["max_depth"]
            # )
            # See https://github.com/Microsoft/LightGBM/issues/907
            params["num_leaves"] = 31
            params["min_data_in_leaf"] = trial.suggest_int(
                "min_data_in_leaf",
                1,
                max(1, int(self.n_samples / params["num_leaves"])),
            )
            params["l2_leaf_reg"] = trial.suggest_loguniform("lambda_l2", 1e-09, 10.0)

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_discrete_uniform(
                    "bagging_temperature", 0.5, 0.95, 0.05
                )
            elif (
                params["bootstrap_type"] == "Bernoulli"
                or params["bootstrap_type"] == "Poisson"
            ):
                params["subsample"] = trial.suggest_uniform("subsample", 0.1, 1)

            return params

        for name, distribution in self.param_distributions.items():
            params[name] = trial._suggest(name, distribution)

        return params


class CatBoostBase(cb.CatBoost):
    def __init__(
        self,
        params: Dict[str, Any],
        refit: bool = False,
        cv: CVType = 5,
        n_trials: int = 20,
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
        study: Optional[study_module.Study] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(params)
        self.refit = refit
        self.cv = cv
        self.n_trials = n_trials
        self.param_distributions = param_distributions
        self.study = study
        self.timeout = timeout

    def fit(
        self,
        X: MultipleDataType,
        y: Optional[TargetDataType] = None,
        cat_features: Optional[FeatureType] = None,
        text_features: Optional[FeatureType] = None,
        pairs: Optional[PairsType] = None,
        sample_weight: SampleWeightType = None,
        group_id: Optional[FeatureType] = None,
        group_weight: Optional[FeatureType] = None,
        subgroup_id: Optional[FeatureType] = None,
        pairs_weight: Optional[FeatureType] = None,
        baseline: Optional[FeatureType] = None,
        use_best_model: Optional[bool] = None,
        eval_set: Optional[cb.Pool] = None,
        verbose: Optional[Union[bool, int]] = None,
        logging_level: Optional[str] = None,
        plot: bool = False,
        column_description: Optional[str] = None,
        verbose_eval: Optional[Union[bool, int]] = None,
        metric_period: Optional[int] = None,
        silent: Optional[bool] = None,
        early_stopping_rounds: Optional[int] = None,
        save_snapshot: Optional[bool] = None,
        snapshot_file: Optional[str] = None,
        snapshot_interval: Optional[int] = None,
        init_model: Optional[str] = None,
    ):
        logger = logging.getLogger(__name__)

        # catboost\core.py
        # CatBoost._prepare_train_params
        train_params = self._prepare_train_params(
            X,
            y,
            cat_features,
            text_features,
            pairs,
            sample_weight,
            group_id,
            group_weight,
            subgroup_id,
            pairs_weight,
            baseline,
            use_best_model,
            eval_set,
            verbose,
            logging_level,
            plot,
            column_description,
            verbose_eval,
            metric_period,
            silent,
            early_stopping_rounds,
            save_snapshot,
            snapshot_file,
            snapshot_interval,
            init_model,
        )

        n_samples = len(X)
        # get_params
        params = train_params["params"]
        eval_name = params.get("loss_function")
        early_stopping_rounds = early_stopping_rounds
        n_estimators = params.get("iterations")

        is_classifier = self._estimator_type == "classifier"
        is_higher_better = _is_higher_better(eval_name)
        cv = check_cv(cv=self.cv, y=y, classifier=is_classifier)

        if self.study is None:
            sampler = samplers.RandomSampler()
            direction = "maximize" if is_higher_better else "minimize"
            self.study = study_module.create_study(direction=direction, sampler=sampler)
        # hyper_parameter tuning
        dataset = cb.Pool(X, label=y)
        objective = _Objective(
            params,
            dataset=dataset,
            cv=cv,
            eval_name=eval_name,
            n_samples=n_samples,
            is_higher_better=is_higher_better,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators=n_estimators,
            param_distributions=self.param_distributions,
        )

        logger.info("Searching the best hyper_parameters")
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        logger.info("Done")

        logger.info("Starting refit")
        if self.refit:
            pass

        return self

    def _refit(self):
        pass

    def predict(
        self,
        data: MultipleDataType,
        prediction_type: str = "RawFormulaVal",
        ntree_start: int = 0,
        ntree_end: int = 0,
        thread_count: int = -1,
        verbose: Optional[bool] = None,
    ):
        return self._predict(
            data,
            "RawFormulaVal",
            ntree_start,
            ntree_end,
            thread_count,
            verbose,
            "predict",
        )


class CatBoostClassifier(CatBoostBase, ClassifierMixin):
    def __init__(
        self,
        params: Dict[str, Any],
        refit: bool = False,
        cv: CVType = 5,
        n_trials: int = 20,
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
        study: Optional[study_module.Study] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            params,
            refit=refit,
            cv=cv,
            n_trials=n_trials,
            param_distributions=param_distributions,
            study=study,
            timeout=timeout,
        )

    def predict_proba(
        self,
        data,
        ntree_start: int = 0,
        ntree_end: int = 0,
        thread_count: int = -1,
        verbose: Optional[bool] = None,
    ):
        return self._predict(
            data,
            "Probability",
            ntree_start,
            ntree_end,
            thread_count,
            verbose,
            "predict_proba",
        )


class CatBoostRegressor(CatBoostBase, RegressorMixin):
    def __init__(
        self,
        params: Dict[str, Any],
        refit: bool = False,
        cv: CVType = 5,
        n_trials: int = 20,
        param_distributions: Optional[Dict[str, distributions.BaseDistribution]] = None,
        study: Optional[study_module.Study] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            params,
            refit=refit,
            cv=cv,
            n_trials=n_trials,
            param_distributions=param_distributions,
            study=study,
            timeout=timeout,
        )
