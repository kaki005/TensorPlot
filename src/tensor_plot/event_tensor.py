import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .dense import BaseTensor


class Entry:
    def __init__(self, index: np.ndarray, count: int):
        self.index: np.ndarray = index
        """(モード, 各モードのインデックス)"""
        self.count: int = count


class Event:
    """class of event at time t"""

    def __init__(self, ndims, entries: list[Entry], t: float, dt: datetime | None = None):
        self.t: float = t
        """event datetime"""
        self.ndims = ndims
        self.entries: list[Entry] = entries
        """(イベント数, イベントの各モードインデックス)"""
        self.mode_counts: list[np.ndarray] = []
        """(モード, モードの各インデックスの出現回数)"""
        self.datetime: datetime | None = dt
        # calc of mode counts
        for i, dim in enumerate(self.ndims):
            count = np.zeros(dim)
            for entry in entries:
                count[entry.index[i]] += 1
            self.mode_counts.append(count)


class EventTensor(BaseTensor):
    def __init__(self, ndims, display_name_list: list[list[str]] | None = None, start_date: datetime | None = None):
        super(EventTensor).__init__()
        self.events: list[Event] = []
        self.start_date: datetime | None = start_date
        self.ndims = ndims
        self.display_name_list: list[list[str]] | None = display_name_list
        """(mode, mode index, display name)"""
        self.t_list = []
        self.mode_titles: list[str] | None = None

    def append(self, event: Event):
        self.events.append(event)
        self.t_list.append(event.t)

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
        if self.display_name_list is not None:
            ax.set_yticklabels(self.display_name_list[0])
            ax.set_zticklabels(self.display_name_list[1])

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
                    if self.display_name_list is None:
                        y.append(float(entry.index[mode]))
                    else:
                        y.append(self.display_name_list[mode][entry.index[mode]])
        plt.scatter(x, y, s=circle_size)
        if self.mode_titles is not None:
            plt.ylabel(self.mode_titles[mode])
        plt.tight_layout()
        plt.savefig(save_path)


def load_event_tensor(pkl_path: str) -> EventTensor:
    with open(pkl_path, "rb") as inp:
        return pickle.load(inp)
