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

if __name__ == '__main__':

    # setup output directory
    output_dir = 'output/'
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        output_dir = sys.argv[idx + 1]
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        print('output dir does not exist')
        sys.exit(0)
    output_dir = os.path.join(output_dir, 'processed')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

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

    all_ret = []

    progress = tqdm.tqdm(total=len(paths))
    for i, p in enumerate(paths):
        img = read_image_rgb(p)
        pipe_time = time.time()
        ret = pipeline_mp(img, **all_args)
        pipe_time = time.time() - pipe_time

        plot_time = time.time()
        plt.subplot(121)
        plt.imshow(draw_boxes(ret['img'], ret['on_window']))
        plt.subplot(122)
        plt.title('#on_windows={}'.format(len(ret['on_window'])))
        plt.imshow(np.sum(ret['im_buffer'], axis=0), cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.colorbar()
        plot_time = time.time() - plot_time
        fig.savefig(os.path.join(output_dir, '{}.jpg'.format(i)))
        plt.gcf().clear()


        # pickle ret
        ret['im_path'] = p
        # ret['all_windows'] = len(ret['all_windows'])
        del ret['img']
        del ret['im_buffer']
        all_ret.append(ret)
        with open(os.path.join(output_dir, 'ret.p'), 'wb') as fret:
            pickle.dump(all_ret, fret)
        # save img with overlaying windows and heatmap

        # print('pipeline time: ', pipe_time)
        # print('plot time: ', plot_time)
        # print('save time: ', save_time)


        progress.update()
