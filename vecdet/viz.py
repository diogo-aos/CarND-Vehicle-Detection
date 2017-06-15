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

    print(len(paths), 'images')

    for p in paths:
        img = read_image_rgb(p)
        ret = pipeline_mp(img, n_procs=4, **all_args)

    # for ret in rets:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_boxes(img, ret['on_window']))
        plt.title('#on_windows={}, #windows={}'.format(len(ret['on_window']), len(ret['all_windows'])))
        plt.subplot(122)
        heatmap = create_heatmap(img, ret['on_window'], threshold=run_config['heatmap_min_threshold'])
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.colorbar()
        plt.show()
