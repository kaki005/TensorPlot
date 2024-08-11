import numpy as np
from tensor_plot import TensorPlot


def sin_func(t: np.ndarray, omega: float):
    return np.sin(omega * t)


def main():
    times = np.arange(1, 200)
    tensors = np.array([[sin_func(times, 0.1 * i * j) for j in range(10)] for i in range(3)])
    tensors = tensors.transpose(2, 1, 0)
    plot = TensorPlot()
    plot.set_alpha(200)
    plot.plot_tensor(tensors, "outputs/test.png")


if __name__ == "__main__":
    main()
