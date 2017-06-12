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

    prefix = get_arg('-n') or 'ret_'
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
    fig = plt.figure()

    store = {'run_config': run_config,
             'processed_frames': []}

    progress = tqdm.tqdm(total=len(paths))
    for i, p in enumerate(paths):
        img = read_image_rgb(p)
        pipe_time = time.time()
        ret = pipeline_mp(img, **all_args)
        pipe_time = time.time() - pipe_time

        # pickle ret
        frame = {'im_path': p, 'on_window': ret['on_window']}
        store['processed_frames'].append(frame)
        with open(os.path.join(output_dir, name), 'wb') as fret:
            pickle.dump(store, fret)

        # print('pipeline time: ', pipe_time)
        # print('save time: ', save_time)

        progress.update()
