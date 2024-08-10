import io

import matplotlib
import numpy as np
import opencv2 as cv
from matplotlib.figure import Figure

matplotlib.use("agg")
from typing import dict, list, tuple

import matplotlib.pyplot as plt


class TensorPlot:
    def plot_tensor(self, tensor: np.ndarray, save_path: str, shift: tuple[int, int] = (10, 10), dpi: int = 100):
        overall_img = np.zeros((4 * dpi, 3 * dpi, 3))
        for i in range(tensor.shape[1]):
            fig: Figure = plt.figure(figsize=(4, 3), dpi=dpi)
            ax = fig.add_subplot(111)  # one graph
            ax.plot(range(tensor.shape[0]), tensor[:, i])
            img_array = self._plt_to_ndarray(fig)
            self._overlay(img_array, overall_img, shift=(shift[0] * i, shift[1] * i))
        cv.imwrite(save_path, overall_img)

    @staticmethod
    def _plt_to_ndarray(fig: Figure, dpi: int = 100) -> np.ndarray:
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        fig.savefig(buf, format="png", dpi=dpi)  # bufに書き込み
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(
            buf.getvalue(), dtype=np.uint8
        )  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする)
        img = cv.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
        return img

    @staticmethod
    def _overlay(fore_img: np.ndarray, back_img: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
        """
        fore_img：合成する画像
        back_img：背景画像
        shift：左上を原点としたときの移動量(x, y)
        """

        shift_x, shift_y = shift
        fore_h, fore_w = fore_img.shape[:2]
        fore_x_min, fore_x_max = 0, fore_w
        fore_y_min, fore_y_max = 0, fore_h

        back_h, back_w = back_img.shape[:2]
        back_x_min, back_x_max = shift_y, shift_y + fore_h
        back_y_min, back_y_max = shift_x, shift_x + fore_w

        if back_x_min < 0:
            fore_x_min = fore_x_min - back_x_min
            back_x_min = 0
        if back_x_max > back_w:
            fore_x_max = fore_x_max - (back_x_max - back_w)
            back_x_max = back_w
        if back_y_min < 0:
            fore_y_min = fore_y_min - back_y_min
            back_y_min = 0
        if back_y_max > back_h:
            fore_y_max = fore_y_max - (back_y_max - back_h)
            back_y_max = back_h

        back_img[back_y_min:back_y_max, back_x_min:back_x_max] = fore_img[fore_y_min:fore_y_max, fore_x_min:fore_x_max]
        return back_img
