import hydra
import numpy as np
import omegaconf
import wandb
from configs import Config
from tensor_plot import TensorPlot


def sin_func(t: np.ndarray, omega: int):
    return np.sin(omega * t)


# @hydra.main(version_base=None, config_path="configs/", config_name="default")
def main(cfg: Config):
    times = np.arange(1, 100)
    tensors = np.array([sin_func(times, i * j) for j in range(3) for i in range(3)])
    tensors = tensors.transpose(2, 1, 0)
    plot = TensorPlot()
    plot.plot_tensor(tensors, "test.png")


if __name__ == "__main__":
    main()
