import numpy as np
import cv2
import sys
import os.path
from utils import *

video_path = get_arg('-v')
output_dir = get_arg('-o')

print(video_path)
print(output_dir)

input('is ok?')

if not video_path:
    print('missing video path -v')
    sys.exit(0)
if not output_dir:
    print('missing output dir -o')
    sys.exit(0)

cap = cv2.VideoCapture(video_path)

i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(i)), frame)
    i += 1

print('finished extracting frames')
