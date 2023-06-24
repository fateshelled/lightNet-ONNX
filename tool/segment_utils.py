import numpy as np
import cv2


COLORS = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]
], dtype=np.uint8)


def colorize(labeled_img):
    return COLORS[labeled_img]


def overlay(img0, img1, alpha=0.5):
    resized = cv2.resize(img1, img0.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(img0, 1 - alpha, resized, alpha, 0)
