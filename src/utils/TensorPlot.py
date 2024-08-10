import opencv2 as cv
import numpy as np
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict




class TensorPlot:

    @staticmethod
    def plt_to_ndarray(fig : Figure) -> np.ndarray:
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


    @staticmethod
    def overlay(fore_img: np.ndarray, back_img : np.ndarray, shift: Tuple[int,int]) -> np.ndarray:
 
        '''
        fore_img：合成する画像
        back_img：背景画像
        shift：左上を原点としたときの移動量(x, y)
        '''
    
        shift_x, shift_y = shift
    
        fore_h, fore_w = fore_img.shape[:2]
        fore_x_min, fore_x_max = 0, fore_w
        fore_y_min, fore_y_max = 0, fore_h
    
        back_h, back_w = back_img.shape[:2]
        back_x_min, back_x_max = shift_y, shift_y+fore_h
        back_y_min, back_y_max = shift_x, shift_x+fore_w
    
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