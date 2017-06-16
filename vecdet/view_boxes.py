import sys
import json
import glob

import cv2
import numpy as np

from lessons import *
from utils import *

if __name__ == '__main__':

    run_config = get_run_config_from_cli(sys.argv)
    if not run_config:
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
        final_im = write_on_image(img_boxes, [str(scaled_window_size)])
        images.append(final_im)

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
            key = cv2.waitKey()
            if key == ord('n'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord('s'):
                fn = input('filename:')
                cv2.imwrite(fn, im)

    cv2.destroyAllWindows()
