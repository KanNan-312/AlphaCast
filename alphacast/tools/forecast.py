# Thin wrapper around the candidate model pool M (alphacast/models/base.py)
# used to produce a single model's forecast for a given look-back window --
# i.e. to materialize Mi(Xen), the per-model prediction that feeds the
# case-based prediction Ycase (paper Eq. 2) -- and to write predictions to
# the CSV format consumed by eval.align_predictions.
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL
from ..models.base import (
    ForecastModel,
    configure_deep_learning_runtime,
    get_default_models,
)
if TYPE_CHECKING:
    from alphacast.config import DatasetConfig
from ..utils.time import generate_future_timestamps


def forecast_with_model(
    model_name: str,
    last_window: np.ndarray,
    h: int,
    season_length: Optional[int],
    dataset: Optional["DatasetConfig"] = None,
    **kwargs
) -> np.ndarray:
    """Fit candidate model `model_name` (Mi(Xen) from the case-library pool)
    on `last_window` and forecast `h` steps ahead. Falls back to
    SeasonalNaive if `model_name` is unknown (e.g. not in get_default_models()).
    `dataset.checkpoints` configures pretrained weights for deep-learning /
    foundation candidate models when applicable."""
    if dataset is not None:
        configure_deep_learning_runtime(dataset.checkpoints, dataset.predicted_window, dataset.frequency)

    models = {m.alias: m for m in get_default_models()}
    if model_name not in models:
        model_name = "SeasonalNaive"
    model = models[model_name]

    timestamps = kwargs.get("timestamps", None)
    future_timestamps = generate_future_timestamps(timestamps.iloc[-1], h, pd.infer_freq(pd.to_datetime(timestamps)))

    # Pass timestamps through to fit; keep predict unchanged
    model.fit(last_window, season_length=season_length, timestamps=timestamps)
    return model.predict(h, future_timestamps=future_timestamps)


def save_predictions_csv(out_path: str, timestamps: List[pd.Timestamp], preds: np.ndarray) -> None:
    """Write a forecast as `time_stamp`/`predicted_ans` rows, the layout
    expected by eval.align_predictions when scoring against ground truth."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({"time_stamp": timestamps, "predicted_ans": preds})
    df.to_csv(out_path, index=False)
