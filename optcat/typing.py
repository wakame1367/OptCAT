from typing import Union, Iterable, List

import catboost as cb
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.model_selection import BaseCrossValidator

CVType = Union[int, Iterable, BaseCrossValidator]
TargetDataType = Union[cb.Pool, np.ndarray, pd.DataFrame, pd.Series]
TwoDimFeatureType = Union[List, pd.DataFrame, pd.Series]
TwoDimSparseType = Union[pd.SparseDataFrame, spmatrix]
MultipleDataType = Union[cb.Pool, TwoDimFeatureType, TwoDimSparseType]
