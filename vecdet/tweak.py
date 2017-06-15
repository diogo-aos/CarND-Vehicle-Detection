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
from scipy.ndimage.measurements import label
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

with open(data_path, 'rb') as f:
    processed = pickle.load(f)
    frames = processed['processed_frames']
    run_config = processed['run_config']
print('loaded {} frames'.format(len(frames)))

timeout = 0
inc = 1
i = 0

n_frames = len(frames)

buf_len = 10
thresh = 7
thresh2 = 2

show_buffer_boxes = False
show_frame_boxes = False
show_heatmap = True

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
        # frameheat = create_heatmap(im, frame['on_window'], threshold=thresh2)
        # bboxes = get_bboxes_from_heatmap(frameheat)
        # print('thresholded boxes:', len(bboxes))
        # windows.extend(bboxes)
        windows.extend(frame['on_window'])
    print(windows)

    heatmap_0 = create_heatmap(im, windows, threshold=0)
    heatmap = create_heatmap(im, windows, threshold=thresh)

    # heatmap metadata
    max_heat = heatmap_0.max()
    nz_mean_heat = heatmap_0[heatmap_0 > 0].mean()

    txt = []
    txt.append('{} win in frame {}'.format(len(current['on_window']), i))
    txt.append('{} win in buffer of size {}'.format(len(windows), len(buf_frames)))
    txt.append('heatmap max={}, nz mean={}'.format(max_heat, nz_mean_heat))
    txt.append('heatmap threshold = {}'.format(thresh))

    # transform heatmap
    heat_disp = (heatmap * 100 / heatmap.max()).astype(np.uint8)
    heat_disp = np.stack((heatmap, heatmap, heatmap), axis=2)
    heat_disp = cv2.resize(heat_disp, (new_w, new_h))
    heat_disp[...,:2] = 0

    if show_buffer_boxes:
        im = draw_boxes(im, windows)

    if show_frame_boxes:
        im = draw_boxes(im, frame['on_window'])

    def combine_boxes(box1, box2):
        left = min(box1[0][0], box2[0][0])
        right = max(box1[1][0], box2[1][0])

        top = min(box1[0][1], box2[0][1])
        bot = max(box1[1][1], box2[1][1])

        return ((left, top), (right, bot))

    def decide_combine(box1, box2, min_dist=30):
        # box1 should be to the left of box2
        if box1[0][0] > box2[0][0]:
            tmp = box1
            box1 = box2
            box2 = tmp

        combine_x = True
        if box2[0][0] - box1[1][0] < min_dist:
            combine_x = True

    def filter_out(box, min_w=20, min_h=20):
        top_left, bot_right = box
        left, top = top_left
        right, bot = bot_right

        if right-left < min_w or bot - top < min_h:
            return False
        return True

    if show_heatmap:
        labels = label(heatmap)
        # im_disp = draw_labeled_bboxes(im_disp, labels)
        boxes = get_labeled_bboxes(labels)
        print(boxes)
        boxes = [box for box in boxes if filter_out(box)]

        txt.append('# cars = {}'.format(labels[1]))
        im = draw_boxes(im, boxes, color=(255, 0, 0))

    for t in txt:
        print(t)

    im = write_on_image(im, txt)

    im_disp = cv2.resize(im, (new_w, new_h))
    cv2.imshow('frame', im_disp)

    key = cv2.waitKey(timeout)

    # exit
    if key == ord('q'):
        break

    # viz controls
    if key == ord('a'):
        show_buffer_boxes = not show_buffer_boxes
    elif key == ord('s'):
        show_frame_boxes = not show_frame_boxes
    elif key == ord('d'):
        show_heatmap = not show_heatmap
    elif key == ord(' '):
        if timeout == 0:
            timeout = 42
        else:
            timeout = 0

    elif key == ord('w'):
        img = cv2.imread(current['im_path'])
        fn = input('filename:')
        cv2.imwrite(fn, img)

    # tweak controls
    if key == ord('y'):
        thresh += 1
    elif key == ord('h'):
        thresh -= 1
    elif key == ord('t'):
        buf_len += 1
    elif key == ord('g'):
        buf_len -= 1

    # frame movement controls
    elif key == ord('m'):
        timeout = 0
        i += inc
        i = min(i, n_frames - 1)
    elif key == ord('n'):
        timeout = 0
        i -= inc
        i = max(0, i)
    elif key == ord('k'):
        timeout = 0
        i += 10
        i = min(i, n_frames - 1)
    elif key == ord('j'):
        timeout = 0
        i -= 10
        i = max(0, i)
    elif key == ord('o'):
        timeout = 0
        i += 100
        i = min(i, n_frames - 1)
    elif key == ord('i'):
        timeout = 0
        i -= 100
        i = max(0, i)

    # slideshow mode
    else:
        i += inc
        i = min(i, n_frames - 1)
        if i == n_frames - 1:
            timeout = 42
cv2.destroyAllWindows()
