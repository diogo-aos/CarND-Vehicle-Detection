import cv2
import sys

from lessons import *

def pipeline(img):
    ...


if __name__ == '__main__':
    fn = sys.argv[1]
    img = cv2.imread(fn)
    ret = pipeline(img)
    cv2.imshow('frame', ret)
    cv2.waitKey()
    cv2.destroyAllWindows()
