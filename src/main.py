import matplotlib
import numpy as np
from tensor_plot import Series, TensorPlot

matplotlib.use("agg")
cmap_names = ["tab20", "tab20b", "tab20c", "Set3"]
cm_colors = []
for cmap_name in cmap_names:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cm_colors += list(cmap.colors)


def sin_func(t: np.ndarray, omega: float):
    return np.sin(omega * t)


def main():
    times = np.arange(1, 150, 0.3)
    tensors = np.array([[sin_func(times, 0.1 * i * j) for j in range(4)] for i in range(5)])
    tpl = TensorPlot()
    for i, series in enumerate(tensors):
        series = Series(times, series.transpose(), linewidth=0.7)
        labels = ["series1", "eries2", "series3", "series4"] if i == len(tensors) - 1 else []
        series.set_legend(labels)
        for j in range(5):
            series.draw_background(j * 30, (j + 1) * 30, cm_colors[i + j], alpha=0.3)
        tpl.add_series(series)
    tpl.set_alpha(220)
    tpl.plot_tensor("outputs/sample2.png")


if __name__ == "__main__":
    main()
