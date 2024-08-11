import numpy as np
from tensor_plot import Series, TensorPlot


def sin_func(t: np.ndarray, omega: float):
    return np.sin(omega * t)


def main():
    times = np.arange(1, 150, 0.3)
    tensors = np.array([[sin_func(times, 0.1 * i * j) for j in range(4)] for i in range(5)])
    plot = TensorPlot()
    for i, series in enumerate(tensors):
        labels = ["series1", "series2", "series3", "series4"] if i == len(tensors) - 1 else []
        plot.add_series(Series(times, series.transpose(), title=f"country {i}", labels=labels, linewidth=0.5))
    plot.set_alpha(220)
    plot.plot_tensor("outputs/test2.png")


if __name__ == "__main__":
    main()
