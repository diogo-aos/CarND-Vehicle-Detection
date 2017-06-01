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


def pipeline(img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             cv_args,
             scaler, classifier,
             *args, **kwargs):
    im_buffer = get_im_buffer(kwargs.get('buffer_max_len'))
    ret = {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scales = np.linspace(window_scale_min_max[0], window_scale_min_max[1],
                         window_n_scales)
    on_window = []
    all_windows = []
    windows_time = time.time()
    for scale in scales:
        scale_on_windows = []
        scaled_window_size = [int(scale * c) for c in window_size]

        windows = slide_window(img, x_start_stop=x_start_end,
                               y_start_stop=y_start_end,
                               xy_window=scaled_window_size,
                               xy_overlap=window_overlap)
        all_windows.extend(windows)
    for window in all_windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))
        test_features = extract_features(test_img, **cv_args)
        test_scaled = scaler.transform(np.array(test_features).reshape(1, -1))
        is_car = classifier.predict(test_scaled)
        if is_car:
            on_window.append(window)

    windows_time = time.time() - windows_time

    heatmap = create_heatmap(img, on_window, threshold=1)
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
        plt.imshow(draw_boxes(ret['img'], ret['on_window']))
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
