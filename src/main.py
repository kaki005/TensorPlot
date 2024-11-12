import matplotlib
import numpy as np
import tensorly.decomposition as decomp
from tensor_plot import DenseTensor, Event, EventTensor, Series, TuckerTensor

matplotlib.use("agg")
cmap_names = ["tab20", "tab20b", "tab20c", "Set3"]
cm_colors = []
for cmap_name in cmap_names:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    cm_colors += list(cmap.colors)


def main():
    times = np.arange(1, 150, 0.3)
    tensors = np.array([[np.sin(0.1 * i * j * times) for j in range(4)] for i in range(5)])
    dense = DenseTensor()
    for i, series in enumerate(tensors):
        series = Series(times, series.transpose(), linewidth=0.7)
        series.set_title(f"country {i}", font_size=10)
        for j in range(5):
            series.draw_background(j * 30, (j + 1) * 30, cm_colors[i + j], alpha=0.3)
        dense.add_series(series)
    # dense.plot_flat("outputs/sample3.png")

    # タッカー分解
    tensor = np.random.random((10, 10, 10))
    core, factors = decomp.tucker(tensor, rank=[5, 5, 5])
    tucker = TuckerTensor(core, factors)
    tucker.plot_flat("outputs/sample4.png")


if __name__ == "__main__":
    main()
