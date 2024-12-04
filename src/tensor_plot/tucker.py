import matplotlib.pyplot as plt
import numpy as np

from tensor_plot.dense import BaseTensor


class TuckerTensor(BaseTensor):
    def __init__(self, core: np.ndarray, factors: list[np.ndarray]):
        super().__init__()
        self.factors: list[np.ndarray] = factors
        self.core: np.ndarray = core
        """core tensor"""

    @property
    def CoreRank(self):
        return self.core.shape

    def plot_flat(self, save_path: str):
        """plot core tensor and factor matrics"""
        fig, axes = plt.subplots(1, len(self.factors) + 1, figsize=(15, 5))
        # コアテンソルの可視化
        axes[0].imshow(self.core[:, :, 0], cmap="viridis")
        axes[0].set_title("Core Tensor (slice)")
        # 因子行列の可視化
        for i, factor in enumerate(self.factors):
            axes[i + 1].imshow(factor, aspect="auto", cmap="viridis")
            axes[i + 1].set_title(f"Factor Matrix {i+1}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
