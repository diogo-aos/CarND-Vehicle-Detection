import sys
from collections import deque
import json
import pickle
import time
import glob
import multiprocessing as mp

import cv2
import numpy as np
import tqdm

from lessons import *
from utils import *

im_buffer = None

def get_im_buffer(max_lex):
    global im_buffer
    if im_buffer is None:
        im_buffer = deque([], max_lex)
    return im_buffer

def process_window(img, cv_args, scaler, classifier,
                    *args, **kwargs):
    test_img = cv2.resize(img, (64, 64))
    test_features = extract_features(test_img, **cv_args)
    test_scaled = scaler.transform(np.array(test_features).reshape(1, -1))
    is_car = classifier.predict(test_scaled)

    return is_car == 1

def process_scale(scale, img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             cv_args,
             scaler, classifier,
             *args, **kwargs):
    scaled_window_size = [int(scale * c) for c in window_size]

    windows = slide_window(img, x_start_stop=x_start_end,
                           y_start_stop=y_start_end,
                           xy_window=scaled_window_size,
                           xy_overlap=window_overlap)
    on_window = []
    for window in windows:
        window_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        is_car = process_window(window_img, cv_args, scaler, classifier,
                                 *args, **kwargs)
        if is_car:
            on_window.append(window)

    return on_window, windows

def pipeline(img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             heatmap_min_threshold, cv_args,
             scaler, classifier, buffer_max_len,
             *args, **kwargs):
    my_args = locals()

    im_buffer = get_im_buffer(buffer_max_len)
    ret = {}  # return dict

    scales = np.linspace(window_scale_min_max[0], window_scale_min_max[1],
                         window_n_scales)
    on_window = []
    all_windows = []
    windows_time = time.time()
    for scale in scales:
        scaled_detections, scaled_windows = process_scale(scale, **my_args)

        on_window.extend(scaled_detections)
        all_windows.extend(scaled_windows)

    ret['img'] = img
    ret['all_windows'] = all_windows
    ret['on_window'] = on_window

    # print('checked {} windows'.format(len(all_windows)))
    # print('took {} seconds to process all windows'.format(round(windows_time, 2)))
    # print('took {} seconds per window'.format(round(windows_time / len(all_windows), 2)))
    return ret


def process_scale_mp(q, scale, img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             heatmap_min_threshold, cv_args,
             scaler, classifier,
             *args, **kwargs):
    start = time.time()
    on_window, windows = process_scale(scale, img, x_start_end, y_start_end, window_size,
                                         window_scale_min_max, window_n_scales, window_overlap,
                                         cv_args,
                                         scaler, classifier,
                                         *args, **kwargs)
    print('inner proc time scale {}: {}'.format(scale, time.time() - start))
    q.put(on_window)

def process_windows_mp(q, window_list, progbar, img, cv_args, scaler, classifier,
                       *args, **kwargs):
    on_window = []
    # start = time.time()
    if progbar:
        progress = tqdm.tqdm(total=len(window_list))
    for window in window_list:
        x0, y0 = window[0]
        x1, y1 = window[1]
        is_car = process_window(img[y0:y1,x0:x1,:], cv_args, scaler, classifier,
                                *args, **kwargs)
        if is_car:
            on_window.append(window)

        if progbar:
            progress.update()
    # print('inner proc time scale: {}'.format(time.time() - start))
    q.put(on_window)

def pipeline_mp(img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             heatmap_min_threshold, cv_args,
             scaler, classifier, buffer_max_len, n_procs=4,
             *args, **kwargs):
    ''' expects RGB img'''
    my_args = locals()
    im_buffer = get_im_buffer(buffer_max_len)
    ret = {}

    scales = np.linspace(window_scale_min_max[0], window_scale_min_max[1],
                         window_n_scales)
    on_window = []
    all_windows = []
    windows_time = time.time()
    procs, queues = [], []
    for scale in scales:
        scaled_window_size = [int(scale * c) for c in window_size]

        windows = slide_window(img, x_start_stop=x_start_end,
                               y_start_stop=y_start_end,
                               xy_window=scaled_window_size,
                               xy_overlap=window_overlap)
        all_windows.extend(windows)

    win_intervals = np.linspace(0, len(all_windows), n_procs + 1, dtype=np.int)

    for i in range(n_procs):
        progbar = False if i != 0 else False
        win_lst = all_windows[win_intervals[i]:win_intervals[i+1]]
        q = mp.Queue()
        p = mp.Process(target=process_windows_mp, args=(q, win_lst, progbar), kwargs=my_args)
        p.start()
        procs.append(p)
        queues.append(q)

    for p, q in zip(procs, queues):
        p.join()
        on_window.extend(q.get())
    windows_time = time.time() - windows_time

    ret['img'] = img
    ret['all_windows'] = all_windows
    ret['on_window'] = on_window

    # print('checked {} windows'.format(len(all_windows)))
    # print('took {} seconds to process all windows'.format(round(windows_time, 2)))
    # print('took {} seconds per window'.format(round(windows_time / len(all_windows), 2)))
    return ret


if __name__ == '__main__':
    run_config = get_run_config_from_cli(sys.argv)
    scaler_classifier, train_config = get_stuff_from_run_config(run_config)

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
        img = read_image_rgb(p)
        ret = pipeline_mp(img, n_procs=4, **all_args)
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
