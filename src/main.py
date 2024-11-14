import matplotlib
import numpy as np
import tensorly.decomposition as decomp
from tensor_plot import DenseTensor, Entry, Event, EventTensor, Series, TuckerTensor

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

    # event tensor
    ndims = np.array([6, 4])
    display_name_list = [[f"mode{i}_{j}" for j in range(dim)] for i, dim in enumerate(ndims)]
    events = EventTensor(ndims, display_name_list)
    events.append(
        Event(
            ndims,
            [
                Entry(np.array([4, 2]), 1),
                Entry(np.array([5, 2]), 2),
                Entry(np.array([0, 1]), 2),
            ],
            3.3,
        )
    )
    events.append(
        Event(
            ndims,
            [
                Entry(np.array([3, 3]), 3),
                Entry(np.array([4, 2]), 5),
                Entry(np.array([0, 1]), 5),
            ],
            3.7,
        )
    )
    events.plot("outputs/sample5.png")
    events.plot_mode(0, "outputs/sample6.png")


if __name__ == "__main__":
    main()
