import sys
from collections import deque
import json
import pickle
import time
import glob
import os
import os.path
import shutil
import datetime

import tqdm
import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from lessons import *
from utils import *
from pipeline_mp import *

if __name__ == '__main__':

    def get_arg(key):
        if key in sys.argv:
            idx = sys.argv.index(key)
            val = sys.argv[idx + 1]
            return val
        return None

    # setup output directory
    output_dir = get_arg('-o') or 'output/'
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        print('output dir does not \exist')
        sys.exit(0)
    # output_dir = os.path.join(output_dir, 'processed')
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    # os.mkdir(output_dir)

    prefix = get_arg('-n') or 'ret'
    now = datetime.datetime.now()
    name = now.strftime('{}_%Y_%m_%d_%H_%M_%S.p'.format(prefix))

    # get and prepare configurations for pipeline
    run_config = get_run_config_from_cli(sys.argv)
    scaler_classifier, train_config = get_stuff_from_run_config(run_config)

    all_args = run_config.copy()
    all_args.update(scaler_classifier)
    all_args['cv_args'] = train_config


    # get files
    def get_path_sort_key(fn):
        base = os.path.basename(fn)  # filename
        base = base.split('.')[0]  # filename w/o extension
        return int(base)

    fn = sys.argv[1]
    paths = glob.glob(fn)
    paths = sorted(paths, key=get_path_sort_key)

    store = {'run_config': run_config,
             'processed_frames': []}

    n_procs = get_arg('-nprocs') or '4'
    n_procs = int(n_procs)


    # import multiprocessing as mp
    # def process_frames(q, frames, progbar, **all_args):
    #     processed = []
    #     if progbar:
    #         progress = tqdm.tqdm(total=len(frames))
    #     for frame_path in frames:
    #         img = read_image_rgb(frame_path)
    #         ret = pipeline(img, **all_args)
    #         frame = {'im_path': p, 'on_window': ret['on_window']}
    #         processed.append(frame)
    #         if progbar:
    #             progress.update()
    #     q.put(processed)
    #

    # procs, queues = [], []
    # path_intervals = np.linspace(0, len(paths), n_procs + 1, dtype=np.int)
    # for i in range(n_procs):
    #     progbar = False if i != 0 else True
    #     path_lst = paths[path_intervals[i]:path_intervals[i+1]]
    #     q = mp.Queue()
    #     p = mp.Process(target=process_frames, args=(q, path_lst, progbar), kwargs=all_args)
    #     p.start()
    #     procs.append(p)
    #     queues.append(q)
    #
    # print('waiting for processes to end...')
    # for p, q in zip(procs, queues):
    #     p.join()
    #     store['processed_frames'].extend(q.get())
    #
    #
    # with open(os.path.join(output_dir, name), 'wb') as fret:
    #     pickle.dump(store, fret)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    progress = tqdm.tqdm(total=len(paths))
    for i, p in enumerate(paths):
        img = read_image_rgb(p)
        pipe_time = time.time()
        ret = pipeline_mp(img, n_procs=n_procs, **all_args)
        pipe_time = time.time() - pipe_time

        # pickle ret
        frame = {'im_path': p, 'on_window': ret['on_window']}
        store['processed_frames'].append(frame)
        with open(os.path.join(output_dir, name), 'wb') as fret:
            pickle.dump(store, fret)

        # print('pipeline time: ', pipe_time)
        # print('save time: ', save_time)

        progress.update()
