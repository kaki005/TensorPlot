import numpy as np


class Series:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fig_size: tuple[int, int] = (12, 3),
        linewidth: float = 1.0,
    ) -> None:
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.fig_size: tuple[int, int] = fig_size
        self.labels: list[str] = []
        self.linewidth: float = linewidth
        self.title = ""
        self.regimes: list[Regime] = []
        self.title_font = 12
        self.ylabel: str | None = None

    def set_title(self, title: str, font_size: int = 12):
        self.title = title
        self.title_font = font_size

    def set_legend(self, labels: list[str]):
        self.labels = labels

    def set_ylabel(self, ylabel: str):
        self.ylabel = ylabel

    def draw_background(
        self,
        x1: float,
        x2: float,
        color,
        alpha: float = 0.5,
        linewidth: float = 0.3,
        fill: bool = True,
        edge_color: str = "white",
    ):
        self.regimes.append(Regime(x1, x2, color, alpha, linewidth, fill, edge_color))


class Regime:
    def __init__(self, x1: float, x2: float, color, alpha: float, linewidth: float, fill: bool, edge_color: str):
        self.x1: float = x1
        self.x2: float = x2
        self.color = color
        self.alpha: float = alpha
        self.linewidth: float = 0.3
        self.fill: bool = fill
        self.edge_color: str = edge_color
