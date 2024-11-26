import pickle
from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.tree_util import register_static
from mpl_toolkits.mplot3d import Axes3D
from pandas import Timestamp
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .dense import BaseTensor


@register_static
class Entry:
    def __init__(self, index: np.ndarray, count: int, t: float):
        self.index: np.ndarray = index
        """(モード, 各モードのインデックス)"""
        self.count: int = count
        """count of index time"""
        self.t: float = t
        """event occurrence time"""


class Event:
    """Set of event entries that occurred at the same time"""

    def __init__(self, ndims, entries: list[Entry], t: float, dt: Timestamp | None = None):
        self.t: float = t
        """event occurrence time"""
        self.ndims = ndims
        """(mode, dimension of mode)"""
        self.entries: list[Entry] = entries
        """list of index"""
        self.datetime: Timestamp | None = dt

    @property
    def count(self) -> int:
        count = 0
        for entry in self.entries:
            count += entry.count
        return count

    @property
    def mode_counts(self) -> list[np.ndarray]:
        """(mode, number of occurrences of each index in the mode)"""
        mode_counts: list[np.ndarray] = []
        for i, dim in enumerate(self.ndims):
            count = np.zeros(dim)
            for entry in self.entries:
                count[entry.index[i]] += 1
            mode_counts.append(count)
        return mode_counts

    @property
    def indexes(self) -> jnp.ndarray:
        """(entry num, index of the entry)"""
        indexes = []
        for entry in self.entries:
            for _ in range(entry.count):
                indexes.append(entry.index)
        return jnp.array(indexes)


class EventTensor(BaseTensor):
    def __init__(self, ndims: np.ndarray, columns: list[list[str]] | None = None, st_date: datetime | None = None):
        super().__init__()
        self.events: list[Event] = []
        self.st_date: datetime | None = st_date
        """start datetime"""
        self.ndims: np.ndarray = ndims
        self.columns: list[list[str]] | None = columns
        """(mode, mode index, display name)"""
        self.mode_titles: list[str] | None = None

    @property
    def tlist(self) -> list[float]:
        """list of event occurene time"""
        return [event.t for event in self.events]

    def append(self, event: Event):
        self.events.append(event)

    def save(self, pkl_path: str):
        with open(pkl_path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def set_titles(self, mode_titles: list[str]):
        self.mode_titles = mode_titles

    def plot(self, save_path: str, marker="o", t_range: list[int] | None = None):
        assert len(self.events) == 2
        plt.clf()  # reset
        fig = plt.figure(figsize=(10, 10))
        ax: Axes3D = Axes3D(fig)
        ax = fig.add_subplot(projection="3d")
        xs, ys, zs = [], [], []
        for event in self.events:
            for entry in event.entries:
                for _ in range(entry.count):
                    xs.append(event.t)
                    ys.append(entry.index[0])
                    zs.append(entry.index[1])

        ax.scatter(xs, ys, zs, marker=marker)
        ax.set_xlabel("time")
        ax.set_yticks(range(self.ndims[0]))
        ax.set_zticks(range(self.ndims[1]))
        if t_range is not None:
            ax.set_xlim(t_range)
        if self.columns is not None:
            ax.set_yticklabels(self.columns[0])
            ax.set_zticklabels(self.columns[1])

        if self.mode_titles is None:
            ax.set_ylabel("mode 0")
            ax.set_zlabel("mode 1")
        else:
            ax.set_ylabel(self.mode_titles[0])
            ax.set_zlabel(self.mode_titles[1])
        plt.savefig(save_path)

    def plot_mode(self, mode: int, save_path: str, circle_size: float = 10.0):
        x = []
        y = []
        plt.clf()  # reset
        for event in self.events:
            for entry in event.entries:
                for _ in range(entry.count):
                    x.append(event.t)
                    if self.columns is None:
                        y.append(float(entry.index[mode]))
                    else:
                        y.append(self.columns[mode][entry.index[mode]])
        plt.scatter(x, y, s=circle_size)
        if self.mode_titles is not None:
            plt.ylabel(self.mode_titles[mode])
        plt.tight_layout()
        plt.savefig(save_path)


def load_event_tensor(pkl_path: str) -> EventTensor:
    with open(pkl_path, "rb") as inp:
        return pickle.load(inp)


def dataframe_to_event_tensor(
    given_data: pd.DataFrame,
    categorical_idxs: list[str],
    time_idx: str,
    freq: str,
    quatntity_idx: str | None,
) -> tuple[EventTensor, OrdinalEncoder, LabelEncoder]:
    """convert pandas.DataFrmae to event_tensor

    Args:
        given_data (pd.DataFrame): DataFrame
        categorical_idxs (list[str]): list of categorical index
        time_idx (str): column of time
        freq (str): _description_

    Returns:
        tuple[EventTensor, OrdinalEncoder,LabelEncoder]: (event_tensor, encoder)
    """
    data = given_data.copy(deep=True)
    data = data.dropna(subset=(categorical_idxs + [time_idx]))
    data = data[[time_idx] + categorical_idxs]

    # Encode timestamps
    data[time_idx] = pd.to_datetime(data[time_idx])
    data[time_idx] = data[time_idx].dt.round(freq)
    data = data.sort_values(time_idx)
    start = data[time_idx].min()
    end = data[time_idx].max()
    ticks = pd.date_range(start, end, freq=freq)
    timepoint_encoder = LabelEncoder()
    timepoint_encoder.fit(ticks)
    data[f"old_{time_idx}"] = data[time_idx]
    data[time_idx] = timepoint_encoder.transform(data[time_idx])

    # Encode categorical features
    oe = OrdinalEncoder()
    data[categorical_idxs] = oe.fit_transform(data[categorical_idxs])
    data[categorical_idxs] = data[categorical_idxs].astype(int)
    data = data.reset_index(drop=True)
    ndims = data[categorical_idxs].max().values + 1
    event_tensors: EventTensor = EventTensor(ndims, oe.categories_, start)
    for dt in data[time_idx].unique():
        current = data[data[time_idx] == dt].reset_index()
        timestamp: Timestamp = current[f"old_{time_idx}"][0]
        rows_num = current.shape[0]
        if rows_num > 1:
            input = [current.iloc[i] for i in range(rows_num)]
            event_tensors.append(rows_to_event(dt, input, categorical_idxs, ndims, quatntity_idx, timestamp))
        else:
            event_tensors.append(
                rows_to_event(dt, [current.iloc[0]], categorical_idxs, ndims, quatntity_idx, timestamp)
            )
    return event_tensors, oe, timepoint_encoder


def rows_to_event(t: float, rows, targets: list[str], ndims, quatntity_idx: str | None, timestamp: Timestamp) -> Event:
    """convert Series column to Event

    Args:
        t (float): current time
        rows (_type_): _description_
        targets (list[str]): list of target column
        ndims (np.ndarray): 各モードの次元

    Returns:
        Event: _description_
    """
    entries: list[Entry] = []
    for row in rows:
        if quatntity_idx is not None:
            entries.append(
                Entry(
                    np.array([row.loc[col] for col in targets]),
                    int(
                        row[quatntity_idx],
                    ),
                    t,
                )
            )
        else:
            entries.append(Entry(np.array([row.loc[col] for col in targets]), 1, t))
    return Event(ndims, entries, t, timestamp)
