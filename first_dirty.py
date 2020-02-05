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
            'beam_width': 1,
            'beam_length': 3,
            'beam_start': 0,
            'beam_round': True,
            'eval_method': {'sectional': False, 'elimination_width': 3, 'max_n': 2, 'elim_double_ends': True},
            'sobel_params': {'ksize':1, 'scale':1, 'delta':0,
                             'ddepth':cv2.CV_64F, 'borderType':cv2.BORDER_DEFAULT}
            }

dc = DonutCorners(**kwargs)

imgs = []
for digit in dg.data:
    dc.fit(digit.reshape(8,8))
    dc.score_all('pydevd' not in sys.modules)
    imgs.append(dc.src)
    imgs.append(dc.scored)

from pickle import dump, load
dump(imgs, open('data/preprocessed2.p', 'wb'))
#show_imgs(imgs)
print(len(dg.target))