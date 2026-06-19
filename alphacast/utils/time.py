from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from ..data_loader import TIME_COL, TARGET_COL


class AnalysisMemory(TypedDict):
    """Summary statistics for a dataset's training series, cached to
    outputs/<dataset>/analysis.json so later stages don't need to
    recompute them across sliding windows."""
    max: float
    min: float
    mean: float
    variance: float
    periodicity_lag: int
    series_length: int
    frequency: Optional[str]


@dataclass
class CaseEntry:
    """One entry in the Case Library (paper Sec 3.3): the z-scored look-back
    window Xi paired with the candidate model Mi that forecast it best."""
    window: List[float]  # z-scored L-length vector
    best_model: str

@dataclass
class ClusterEntry:
    """A cluster Cm from the case-library clustering (Sec 3.3/A.2): `window`
    is the cluster center cm, and `best_model` tallies how often each
    candidate model Mi was the top performer for members of this cluster
    (used to weight the averaged case-based prediction Ycase, Eq. 2)."""
    window: List[float]  # z-scored L-length vector
    best_model: Dict[str, int]  # model_name -> weight
    total_weight: int

@dataclass
class CaseNeighbor:
    """A single retrievable (Xnb, Ynb) pair: a historical look-back window and
    its corresponding prediction, used for nearest-neighbor retrieval (Sec 3.4)."""
    look_back_window: List[float]  # z-scored L-length vector
    pred_window: List[float]

def estimate_periodicity(y: np.ndarray, max_lag: Optional[int] = None) -> int:
    """Estimate the dominant periodicity of `y` via autocorrelation: the lag
    (up to `max_lag`) with the highest autocorrelation. Used as a seasonal
    hint when no explicit frequency/period is configured."""
    if len(y) < 3:
        return 1
    if max_lag is None:
        max_lag = max(2, len(y) // 2)
    y = y - np.mean(y)
    autocorr = np.correlate(y, y, mode="full")[len(y)-1:len(y)-1+max_lag]
    if len(autocorr) < 2:
        return 1
    lag = int(np.argmax(autocorr[1:]) + 1)
    return max(1, lag)


def generate_future_timestamps(last_ts: pd.Timestamp, h: int, freq: Optional[str]) -> List[pd.Timestamp]:
    """Generate the `h` timestamps T immediately following `last_ts`, used to
    label the forecast window so predictions can be aligned with ground
    truth via TIME_COL during evaluation."""
    if freq is None:
        # Fallback: assume uniform daily spacing and increment by i
        return [last_ts + pd.Timedelta(days=i) for i in range(1, h + 1)]
    return list(pd.date_range(start=last_ts, periods=h + 1, freq=freq)[1:])