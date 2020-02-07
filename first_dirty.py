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
            'beam_width': 1.5,
            'fork_spread': 1,
            'beam_length': 4,
            'beam_start': 0.5,
            'beam_round': True,
            'grid_size': 4,
            'search_args': dict(img_shape=(8,8), min_grid=0.1, top_n=2),
            'eval_method': {'sectional': True, 'elimination_width': 2, 'max_n': 2, 'elim_double_ends': True},
            }

dc = DonutCorners(**kwargs)

import cProfile
prof = cProfile.run('imgs = dc.transform(dg.data)')

from pickle import dump, load
dump(imgs, open('data/preprocessed4.p', 'wb'))
#show_imgs(imgs)
print(imgs[:,-14:])