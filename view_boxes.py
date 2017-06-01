import sys
import json
import glob

import cv2
import numpy as np

from lessons import *

if __name__ == '__main__':

    if '--run-config' in sys.argv:
        idx = sys.argv.index('--run-config')
        run_config_fn = sys.argv[idx + 1]
        with open(run_config_fn, 'r') as f:
            run_config = json.load(f)
    else:
        print('missing --run-config')
        sys.exit(0)


def boxes(img, x_start_end, y_start_end, window_size,
             window_scale_min_max, window_n_scales, window_overlap,
             *args, **kwargs):
    images = []
    scales = np.linspace(window_scale_min_max[0], window_scale_min_max[1],
                         window_n_scales)
    n_windows = 0
    for scale in scales:
        scale_on_windows = []
        scaled_window_size = [int(scale * c) for c in window_size]
        # print(scaled_window_size)
        windows = slide_window(img, x_start_stop=x_start_end,
                               y_start_stop=y_start_end,
                               xy_window=scaled_window_size,
                               xy_overlap=window_overlap)
        n_windows += len(windows)
        img_boxes = draw_boxes(img, windows)
        images.append(img_boxes)

    print('{} windows generated'.format(n_windows))
    return images


if __name__ == '__main__':

    fn = sys.argv[1]
    paths = glob.glob(fn)
    for p in paths:
        img = cv2.imread(p)
        ret = boxes(img, **run_config)
        for im in ret:

            cv2.imshow('frame', im)
            while cv2.waitKey() != ord('q'):
                pass

    cv2.destroyAllWindows()
