import sys
from collections import deque
import json
import pickle
import time
import glob
import os
import os.path
import shutil

import tqdm
import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from lessons import *
from utils import *
from pipeline_mp import *

if '-w' in sys.argv:
    idx = sys.argv.index('-w')
    data_path = sys.argv[idx + 1]
else:
    print('missing data path')
    sys.exit(0)

frames = []
with open(data_path, 'rb') as f:
    try:
        while True:
            frame = pickle.load(f)
            frames.append(frame)
    except EOFError:
        pass
print('loaded {} frames'.format(len(frames)))


timeout = 0
inc = 1
i = 0
thresh = 20
buf_len = 5
n_frames = len(frames)

while True:
    current = frames[i]
    im = cv2.imread(current['im_path'])
    h, w, c = im.shape
    new_h, new_w = int(h/2), int(w/2)


    # create heatmap
    left_i = max(0, i - buf_len + 1)
    buf_frames = frames[left_i:i+1]
    windows = []
    for frame in buf_frames:
        windows.extend(frame['on_window'])

    heatmap = create_heatmap(im, windows, threshold=thresh)
    max_heat = heatmap.max()
    nz_mean_heat = heatmap[heatmap > 0].mean()


    print('frame {} has {} windows'.format(i, len(current['on_window'])))

    print('heatmap with')
    print('\t{} frames'.format(len(buf_frames)))
    print('\t{} windows'.format(len(windows)))

    print('heatmap max={}, nz mean={}'.format(max_heat, nz_mean_heat))

    # transform heatmap
    heat_disp = (heatmap * 255 / heatmap.max()).astype(np.uint8)
    heat_disp = np.stack((heatmap, heatmap, heatmap), axis=2)
    heat_disp = cv2.resize(heat_disp, (new_w, new_h))

    im_boxes = draw_boxes(im, windows)
    im_disp = cv2.resize(im, (new_w, new_h))

    alpha = 0.5
    overlay = im_disp.copy()
    overlay[heat_disp != 0] = heat_disp[heat_disp != 0]
    cv2.addWeighted(overlay, alpha, im_disp, 1 - alpha, 0, im_disp)

    # disp = np.vstack((im_disp, heat_disp))
    # print(disp.shape)
    cv2.imshow('frame', im_disp)

    key = cv2.waitKey(timeout)

    if key == ord('q'):
        break
    elif key == ord(' '):
        if timeout == 0:
            timeout = 42
        else:
            timeout = 0
    elif key == ord('m'):
        timeout = 0
        i += inc
        i = min(i, n_frames - 1)
    elif key == ord('n'):
        timeout = 0
        i -= inc
        i = max(0, i)
    else:
        i += inc
        i = min(i, n_frames - 1)
cv2.destroyAllWindows()
