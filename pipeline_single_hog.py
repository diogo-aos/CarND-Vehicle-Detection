import sys
from collections import deque
import json
import pickle
import time
import glob

import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from lessons import *
from utils import *

im_buffer = None

def get_im_buffer(max_lex):
    global im_buffer
    if im_buffer is None:
        im_buffer = deque([], max_lex)
    return im_buffer

def pipeline_scale(img_search, scale, window_overlap,
             cv_args,
             scaler, classifier,
             *args, **kwargs):
    img = img_search
    print(img.shape)
    if scale != 1:
        # scale the image so the windows are 64x64
        h, w, c = img.shape
        img = cv2.resize(img, (np.int(w / scale), np.int(h / scale)))
    print(img.shape)
    h, w, c = img.shape

    pix_per_cell = cv_args['pix_per_cell']
    cell_per_block = cv_args['cell_per_block']
    orient = cv_args['orient']
    hog_channel = cv_args['hog_channel']

    # Define blocks and steps as above
    nxblocks = (w // pix_per_cell) - cell_per_block + 1
    nyblocks = (h // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    window = 64  # original sample size
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = np.int((1 - window_overlap[0]) * pix_per_cell)
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # compute hogs

    hogs = []
    if hog_channel == 'ALL':
        for channel in range(c):
            hog = get_hog_features(img[...,channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hogs.append(hog)
    else:
        hog = get_hog_features(img[...,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hogs.append(hog)

    all_windows = []
    on_window = []

    spatial_size = cv_args['spatial_size']
    hist_bins = cv_args['hist_bins']

    for yb in range(nysteps):
        for xb in range(nxsteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_features = []
            for hog in hogs:
                h = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features.append(h)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            subimg = cv2.resize(img[ytop:ytop+window, xleft:xleft+window], (64, 64))

            # cv2.imshow('frame', subimg)
            # cv2.waitKey()

            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            features = [spatial_features, hist_features]
            features.extend(hog_features)
            features = np.hstack(features)

            test_features = scaler.transform(features.reshape(1, -1))
            is_car = classifier.predict(test_features)

            # compute window
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            top_left = (xbox_left, ytop_draw)
            bot_right = (xbox_left + win_draw, ytop_draw + win_draw)
            all_windows.append((top_left, bot_right))

            if is_car:
                on_window.append((top_left, bot_right))
    return on_window, all_windows


def pipeline(img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             cv_args,
             scaler, classifier,
             *args, **kwargs):
    my_args = locals()
    im_buffer = get_im_buffer(kwargs.get('buffer_max_len'))
    ret = {}
    img = img.astype(np.float32)/255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_crop = img[y_start_end[0]:y_start_end[1], x_start_end[0]:x_start_end[1],:]


    scales = np.linspace(window_scale_min_max[0], window_scale_min_max[1],
                         window_n_scales)
    on_window = []
    all_windows = []
    windows_time = time.time()
    for scale in scales:
        scaled_detections, scaled_windows = pipeline_scale(img_crop, scale, **my_args)
        on_window.extend(scaled_detections)
        all_windows.extend(scaled_windows)

    windows_time = time.time() - windows_time
    print(on_window)

    def position_window(window):
        top_left, bot_right = window
        left, top = top_left
        left += x_start_end[0]
        top += y_start_end[0]
        top_left = (left, top)

        right, bot = bot_right
        right += x_start_end[0]
        bot += y_start_end[0]
        bot_right = (right, bot)
        return (top_left, bot_right)

    for i, window in enumerate(on_window):
        on_window[i] = position_window(window)

    all_windows = [position_window(win) for win in all_windows]

    heatmap = create_heatmap(img, on_window, threshold=0)
    im_buffer.appendleft(heatmap)

    ret['img'] = img
    ret['im_buffer'] = list(im_buffer)
    ret['all_windows'] = all_windows
    ret['on_window'] = on_window

    # print('checked {} windows'.format(len(all_windows)))
    # print('took {} seconds to process all windows'.format(round(windows_time, 2)))
    # print('took {} seconds per window'.format(round(windows_time / len(all_windows), 2)))
    return ret


if __name__ == '__main__':
    train_config = get_train_config_from_cli(sys.argv)
    run_config = get_run_config_from_cli(sys.argv)

    with open(run_config['scaler_classifier_path'], 'rb') as f:
        scaler_classifier = pickle.load(f)


    all_args = run_config.copy()
    all_args.update(scaler_classifier)
    all_args['cv_args'] = train_config

    def get_path_sort_key(fn):
        return int(fn.split('/')[-1].split('.')[0])

    fn = sys.argv[1]
    paths = glob.glob(fn)
    paths = sorted(paths, key=get_path_sort_key)
    rets = []

    for p in paths:
        img = cv2.imread(p)
        ret = pipeline(img, **all_args)
        rets.append(ret)

    # for ret in rets:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_boxes(ret['img'], ret['all_windows']))
        plt.title('#on_windows={}, #windows={}'.format(len(ret['on_window']), len(ret['all_windows'])))
        plt.subplot(122)
        plt.imshow(ret['im_buffer'][0], cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.colorbar()
        plt.show()



            # cv2.imshow('frame', im)
            # while cv2.waitKey() != ord('q'):
            #     pass



    # cv2.destroyAllWindows()
