import pickle
import sys

import cv2
import tqdm

from utils import *
from lessons import *

input_fn = get_arg('-i')
video_fn = get_arg('-o')

buf_len = int(get_arg('-buflen'))
thresh = int(get_arg('-thresh'))

with open(input_fn, 'rb') as f:
    processed = pickle.load(f)
    frames = processed['processed_frames']
    run_config = processed['run_config']
print('loaded {} frames'.format(len(frames)))

im = cv2.imread(frames[0]['im_path'])
h, w = im.shape[:2]

fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
video = cv2.VideoWriter(video_fn, fourcc, 24, (w, h))

progress = tqdm.tqdm(total=len(frames))
for i, current in enumerate(frames):
    im = cv2.imread(current['im_path'])
    h, w, c = im.shape

    # create heatmap
    left_i = max(0, i - buf_len + 1)
    buf_frames = frames[left_i:i+1]
    windows = []
    for frame in buf_frames:
        windows.extend(frame['on_window'])
    heatmap = create_heatmap(im, windows, threshold=thresh)

    def filter_out(box, min_w=20, min_h=20):
        top_left, bot_right = box
        left, top = top_left
        right, bot = bot_right

        if right-left < min_w or bot - top < min_h:
            return False
        return True


    labels = label(heatmap)
    boxes = get_labeled_bboxes(labels)
    boxes = [box for box in boxes if filter_out(box)]
    im = draw_boxes(im, boxes, color=(255, 0, 0))
    video.write(im)

    progress.update()

video.release()
