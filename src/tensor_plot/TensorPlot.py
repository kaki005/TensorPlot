import io
from typing import Dict, List, Tuple

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image as im
from PIL.Image import Image

colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
matplotlib.use("agg")


class TensorPlot:
    def plot_tensor(self, tensor: np.ndarray, save_path: str, shift: tuple[int, int] = (50, 50), dpi: int = 100):
        fig_size = (6, 3)
        series_num = tensor.shape[1]
        overall_img = im.new(
            "RGBA",
            (fig_size[0] * dpi + shift[0] * series_num, fig_size[1] * dpi + shift[1] * series_num),
            (255, 255, 255, 255),
        )
        for i in range(series_num):
            fig: Figure = plt.figure(figsize=fig_size, dpi=dpi)
            ax = fig.add_subplot(111)  # one graph
            ax.plot(range(tensor.shape[0]), tensor[:, i])
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            fig.patch.set_alpha(0)  # make figure's background tranparent
            image = self._plt_to_image(fig)
            image = image.convert("RGBA")
            plt.close()
            shift = (shift[0] * i, shift[1] * i)
            overall_img = self._overlay(image, overall_img, shift)
        overall_img.save(save_path)

    @staticmethod
    def _plt_to_image(fig: Figure, dpi: int = 100) -> Image:
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        fig.savefig(buf, format="png", dpi=dpi)  # bufに書き込み
        buf.seek(0)  # ストリーム位置を先頭に戻る
        img_arr = np.frombuffer(
            buf.getvalue(), dtype=np.uint8
        )  # メモリからバイナリデータを読み込み, numpy array 形式に変換
        buf.close()  # ストリームを閉じる(flushする)
        img = cv.imdecode(img_arr, -1)  # 画像のバイナリデータを復元する
        img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
        return im.fromarray(img)

    @staticmethod
    def _overlay(fore_img: Image, back_img: Image, shift: tuple[int, int]) -> Image:
        """
        fore_img：合成する画像
        back_img：背景画像
        shift：左上を原点としたときの移動量(x, y)
        """
        composite_img = im.new("RGBA", back_img.size, (255, 255, 255, 0))
        composite_img.paste(fore_img, shift, fore_img)
        result_image = im.alpha_composite(back_img, composite_img)
        return result_image
        # shift_x, shift_y = shift
        # fore_h, fore_w = fore_img.shape[:2]
        # fore_x_min, fore_x_max = 0, fore_w
        # fore_y_min, fore_y_max = 0, fore_h

        # back_h, back_w = back_img.shape[:2]
        # back_x_min, back_x_max = shift_y, shift_y + fore_w
        # back_y_min, back_y_max = shift_x, shift_x + fore_h
        # print(f"back_x=[{back_x_min}, {back_x_max}]")
        # print(f"back_y=[{back_y_min}, {back_y_max}]")
        # print(f"fore_x=[{fore_x_min}, {fore_x_max}]")
        # print(f"fore_y=[{fore_y_min}, {fore_y_max}]")
        # back_img[back_y_min:back_y_max, back_x_min:back_x_max, mask] = fore_img[fore_y_min:fore_y_max, fore_x_min:fore_x_max, mask]
        # return back_img
