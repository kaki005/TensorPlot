import io
from typing import Dict, List, Tuple

import cv2 as cv
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image as im
from PIL import ImageDraw
from PIL.Image import Image

from .Series import Regime, Series


class TensorPlot:
    def __init__(self) -> None:
        self._alpha = 255
        self.series_list: list[Series] = []
        """transparency in chart.(0-255)"""

    # ======================
    # public
    # ======================
    def set_alpha(self, alpha: int):
        """set transparency in chart.(0-255)"""
        assert alpha >= 0 and alpha < 255
        self._alpha = alpha

    def add_series(self, series: Series):
        self.series_list.append(series)

    def plot_tensor(self, save_path: str, shift: tuple[int, int] = (30, 30), dpi: int = 100):
        overall_w, overall_h = 0, 0
        series_num = len(self.series_list)
        for i, series in enumerate(self.series_list):
            start_point = (shift[0] * (series_num - i), shift[1] * i)
            overall_w = max(overall_w, series.fig_size[0] * dpi + start_point[0] + 100)
            overall_h = max(overall_h, series.fig_size[1] * dpi + start_point[1] + 100)
        overall_img = im.new(
            "RGBA",
            (overall_w, overall_h),
            (255, 255, 255, 255),
        )
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        # print("overall_img =", overall_img.size)
        for i, series in enumerate(self.series_list):
            fig: Figure = plt.figure(figsize=series.fig_size, dpi=dpi)
            ax = fig.add_subplot(111)  # one graph
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            if len(series.labels) > 0:
                for y, label in zip(series.y.transpose(), series.labels, strict=False):
                    ax.plot(series.x, y, label=label, linewidth=series.linewidth)
                ax.legend()
            else:
                ax.plot(series.x, series.y, linewidth=series.linewidth)
            if len(series.regimes) > 0:
                for regime in series.regimes:
                    self._draw_background(ax, regime)
            ax.set_xlim(series.x.min(), series.x.max())
            if series.title != "":
                ax.set_title(series.title, fontsize=series.title_font)
            # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)
            fig.patch.set_alpha(0)  # make figure's background tranparent
            plt.tight_layout()

            image = self._plt_to_image(fig)
            plt.close()

            image = image.convert("RGBA")
            if self._alpha != 255:
                image = self._transparent(image)
            start_point = (shift[0] * (series_num - i), shift[1] * i)
            overall_img = self._overlay(image, overall_img, start_point)
        # draw = ImageDraw.Draw(overall_img)
        # draw.line((shift[0]*series_num, 0, 0, shift[1]*series_num),fill=(255, 255, 0), width=3)
        overall_img.save(save_path)

    # ======================
    # private
    # ======================
    @staticmethod
    def _draw_background(ax: Axes, regime: Regime):
        r = patches.Rectangle(
            xy=(regime.x1, ax.get_ylim()[0]),
            width=(regime.x2 - regime.x1),
            height=abs(ax.get_ylim()[0]) + ax.get_ylim()[1],
            fc=regime.color,
            ec=regime.edge_color,
            linewidth=0.1,
            fill=regime.fill,
            alpha=regime.alpha,
        )
        ax.add_patch(r)

    def _select_color(self, color):
        mean = np.array(color).mean(axis=0)
        return (255, 255, 255, self._alpha) if mean >= 250 else color

    def _transparent(self, src: Image) -> Image:
        src = src.convert("RGBA")
        w, h = src.size
        for y in range(h):
            for x in range(w):
                src.putpixel((x, y), self._select_color(src.getpixel((x, y))))
        return src

    @staticmethod
    def _plt_to_image(fig: Figure, dpi: int = 100) -> Image:
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        buf = io.BytesIO()  # インメモリのバイナリストリームを作成
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=1)  # bufに書き込み
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
