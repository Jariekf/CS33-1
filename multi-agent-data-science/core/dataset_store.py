from uuid import uuid4

import pandas as pd


_DATASET_STORE: dict[str, pd.DataFrame] = {}


def put_dataset(df: pd.DataFrame, dataset_id: str | None = None) -> str:
    final_dataset_id = dataset_id or f"dataset_{uuid4().hex}"
    _DATASET_STORE[final_dataset_id] = df.copy()
    return final_dataset_id


def get_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in _DATASET_STORE:
        raise KeyError(f"dataset_id 不存在: {dataset_id}")
    return _DATASET_STORE[dataset_id]


def update_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    if dataset_id not in _DATASET_STORE:
        raise KeyError(f"dataset_id 不存在: {dataset_id}")
    _DATASET_STORE[dataset_id] = df.copy()
