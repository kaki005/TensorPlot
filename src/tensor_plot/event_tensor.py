import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .dense import BaseTensor


class Entry:
    def __init__(self, index: np.ndarray):
        self.index: np.ndarray = index
        """(モード, 各モードのインデックス)"""


class Event:
    """class of event at time t"""

    def __init__(self, entries: list[Entry], counts: list[np.ndarray], t: float, dt: datetime):
        self.t: float = t
        """event datetime"""
        self.entries: list[Entry] = entries
        """(イベント数, イベントの各インデックス)"""
        self.counts: list[np.ndarray] = counts
        """(モード, モードの各インデックスの出現回数)"""
        self.datetime: datetime = dt


class EventTensor(BaseTensor):
    def __init__(self, start_date: datetime, ndims, dic: dict[str, dict[str, int]]):
        super(EventTensor).__init__()
        self.events: list[Event] = []
        self.start_date: datetime = start_date
        self.ndims = ndims
        self.dic: dict[str, dict[str, int]] = dic
        """(key:column_name, (original_value, index))
        """
        self.t_list = []

    def append(self, event: Event):
        self.events.append(event)
        self.t_list.append(event.t)

    def save(self, pkl_path: str):
        with open(pkl_path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def plot(self, save_path: str, marker="o"):
        assert len(self.events) == 2
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        xs, ys, zs = [], [], []
        for event in self.events:
            xs.append(event.t)
            ys.append(event.entries[0].index)
            zs.append(event.entries[1].index)

        ax.scatter(xs, ys, zs, marker=marker)
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        plt.savefig(save_path)


def load_event_tensor(pkl_path: str) -> EventTensor:
    with open(pkl_path, "rb") as inp:
        return pickle.load(inp)
