import numpy as np
import sys
import cv2
sys.path.append('..')
from donut_corners.donut_corners import DonutCorners

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

def show_img(img, cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

def show_imgs(imgs):
    if len(imgs) == 1:
        show_img(imgs[0])
        return
    
    fig, axs = plt.subplots(ncols=len(imgs))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i], cmap="binary")
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

dg = load_digits()

kwargs = {'angle_count': 12 * 1, # must be multiple of 4
            'beam_count': 12 * 1,
            'beam_width': 2,
            'fork_spread': 1,
            'beam_length': 3,
            'beam_start': 0,
            'beam_round': True,
            'search_args': dict(img_shape=(8,8), edge_offset = 1, top_n = 2, max_rounds = 10, max_step=4, initial_simplex_size=3),
            'eval_method': {'sectional': False, 'elimination_width': 2, 'max_n': 2, 'elim_double_ends': True},
            }

dc = DonutCorners(**kwargs)

imgs = dc.transform(dg.data)

from pickle import dump, load
dump(imgs, open('data/preprocessed4.p', 'wb'))
#show_imgs(imgs)
print(imgs[:,-6:])