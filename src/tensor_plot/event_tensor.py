import pickle
from datetime import datetime
from typing import cast

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
    """
    Represents a single event entry within a tensor.

    Attributes:
        index (np.ndarray): Index values for the event entry (one per mode).
        count (int): The count of this entry.
        t (float): The time of event occurrence.
    """

    def __init__(self, index: np.ndarray, count: int, t: float):
        self.index: np.ndarray = index
        """Index values for the event entry (one per mode)."""
        self.count: int = count
        """The count of this entry."""
        self.t: float = t
        """ The time of event occurrence."""


class Event:
    """
    A collection of entries that occurred at the same time.

    Attributes:
        t (float): Event occurrence time.
        ndims (list[int]): Dimensionality of each mode.
        entries (list[Entry]): List of event entries.
        datetime (Timestamp | None): Optional timestamp for the event.
    """

    def __init__(self, ndims, entries: list[Entry], t: float, dt: Timestamp | None = None):
        self.t: float = t
        """ Event occurrence time."""
        self.ndims = ndims
        """Dimensionality of each mode."""
        self.entries: list[Entry] = entries
        """List of event entries."""
        self.datetime: Timestamp | None = dt
        """Optional event occurence timestamp."""

    @property
    def count(self) -> int:
        """Calculates the total count (weight) across all entries."""
        return sum(entry.count for entry in self.entries)

    @property
    def mode_counts(self) -> list[np.ndarray]:
        """
        Computes the counts for each mode.

        Returns:
            list[np.ndarray]: Count of occurrences for each index in every mode.
        """
        mode_counts = []
        for i, dim in enumerate(self.ndims):
            count = np.zeros(dim)
            for entry in self.entries:
                count[entry.index[i]] += 1
            mode_counts.append(count)
        return mode_counts

    @property
    def indexes(self) -> jnp.ndarray:
        """
        Creates an array of indexes representing all occurrences of this event.

        Returns:
            jnp.ndarray: Array of entry indexes.
        """
        indexes = []
        for entry in self.entries:
            indexes.extend([entry.index] * entry.count)
        return jnp.array(indexes)


class EventTensor(BaseTensor):
    """
    A tensor structure for storing and managing event data.

    Attributes:
        ndims (np.ndarray): Dimensionality of each mode.
        columns (list[list[str]] | None): Optional column names for each mode.
        st_date (datetime | None): Start datetime for the tensor data.
        events (list[Event]): List of events in the tensor.
        mode_titles (list[str] | None): Titles for each mode (used in visualization).
    """

    def __init__(self, ndims: np.ndarray, columns: list[list[str]] | None = None, st_date: datetime | None = None):
        super().__init__()
        self.events: list[Event] = []
        """List of events in the tensor."""
        self.st_date: datetime | None = st_date
        """Start datetime for the tensor data."""
        self.ndims: np.ndarray = ndims
        """Dimensionality of each mode."""
        self.columns: list[list[str]] | None = columns
        """Optional column names for each mode."""
        self.mode_titles: list[str] | None = None
        """Titles for each mode (used in visualization)."""

    @property
    def tlist(self) -> list[float]:
        """Returns a list of event occurrence times."""
        return [event.t for event in self.events]

    @property
    def timestamps(self) -> list[Timestamp]:
        """
        Returns timestamps for events.

        Raises:
            Exception: If datetime information is missing.
        """
        stamps = []
        for event in self.events:
            if event.datetime is None:
                raise Exception("Datetime is not set.")
            stamps.append(event.datetime)
        return stamps

    def append(self, event: Event):
        """Adds a new event to the tensor."""
        self.events.append(event)

    def save(self, pkl_path: str):
        """Serializes the tensor to a pickle file."""
        with open(pkl_path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def set_titles(self, mode_titles: list[str]):
        """Sets display titles for modes."""
        self.mode_titles = mode_titles

    def plot(self, save_path: str, marker="o", t_range: list[int] | None = None):
        """
        Creates a 3D scatter plot of the events.

        Args:
            save_path (str): Path to save the plot.
            marker (str): Marker style for the plot.
            t_range (list[int] | None): Time range for the x-axis.
        """
        assert len(self.events) == 2
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        ax: Axes3D = Axes3D(fig)
        ax = fig.add_subplot(projection="3d")

        xs, ys, zs = [], [], []
        for event in self.events:
            for entry in event.entries:
                xs.extend([event.t] * entry.count)
                ys.extend([entry.index[0]] * entry.count)
                zs.extend([entry.index[1]] * entry.count)

        ax.scatter(xs, ys, zs, marker=marker)
        ax.set_xlabel("Time")
        ax.set_yticks(range(self.ndims[0]))
        ax.set_zticks(range(self.ndims[1]))
        if t_range:
            ax.set_xlim(t_range)
        if self.columns:
            ax.set_yticklabels(self.columns[0])
            ax.set_zticklabels(self.columns[1])
        ax.set_ylabel(self.mode_titles[0] if self.mode_titles else "Mode 0")
        ax.set_zlabel(self.mode_titles[1] if self.mode_titles else "Mode 1")

        plt.savefig(save_path)

    def plot_mode(self, mode: int, save_path: str, circle_size: float = 10.0):
        """
        Plots events for a specific mode.

        Args:
            mode (int): Mode to plot.
            save_path (str): Path to save the plot.
            circle_size (float): Size of the points.
        """
        x, y = [], []
        plt.clf()
        for event in self.events:
            for entry in event.entries:
                x.extend([event.t] * entry.count)
                y.extend(
                    [self.columns[mode][entry.index[mode]] if self.columns else float(entry.index[mode])] * entry.count
                )
        plt.scatter(x, y, s=circle_size)
        plt.ylabel(self.mode_titles[mode] if self.mode_titles else "")
        plt.tight_layout()
        plt.savefig(save_path)


def load_event_tensor(pkl_path: str) -> EventTensor:
    with open(pkl_path, "rb") as inp:
        return cast(EventTensor, pickle.load(inp))


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
    data, oe, timepoint_encoder = encode_dataframe(given_data, categorical_idxs, time_idx, freq, quatntity_idx)
    start = data[time_idx].min()
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


def encode_dataframe(
    given_data: pd.DataFrame,
    categorical_idxs: list[str],
    time_idx: str,
    freq: str,
    quatntity_idx: str | None,
) -> tuple[pd.DataFrame, OrdinalEncoder, LabelEncoder]:
    """DataFrameのtime_idxをTimeStampにエンコードし、categorical_idxsの各列を0,1,2にエンコードします。

    Args:
        given_data (pd.DataFrame): _description_
        categorical_idxs (list[str]): _description_
        time_idx (str):
        freq (str): _description_
        quatntity_idx (str | None): _description_

    Returns:
        tuple[pd.DataFrame, OrdinalEncoder, LabelEncoder]: encoded DataFrame and encoder.
    """
    data = given_data.copy(deep=True)
    target_col = categorical_idxs + [time_idx]
    if quatntity_idx is not None:
        target_col += [quatntity_idx]
    data = data.dropna(subset=(target_col))
    data = data[target_col]

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
    return data, oe, timepoint_encoder
