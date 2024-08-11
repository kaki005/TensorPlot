from typing import Optional

import numpy as np


class Series:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: list[str] | None = None,
        title: str = "",
        fig_size: tuple[int, int] = (12, 3),
        linewidth: float = 1.0,
    ) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.fig_size: tuple[int, int] = fig_size
        self.labels: list[str] = labels if labels is not None else []
        self.linewidth: float = linewidth
        self.title: str = title
