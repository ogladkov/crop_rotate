import cv2
import numpy as np
import os


def crop(img_fname, msk_fname):
    if not os.path.exists('./cropped'):
        os.mkdir('./cropped')

    img = cv2.imread(img_fname)
    msk = cv2.imread(msk_fname, 0)

    ret, thresh = cv2.threshold(msk, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    msk = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(msk)
    box = np.int0(box)
    pts = []
    for p in box:
        pt = [p[0], p[1]]
        pts.append(pt)

    dst = np.array([
        [0, 0],
        [img.shape[1], 0],
        [img.shape[1], img.shape[0]],
        [0, img.shape[0]]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    fout = f'./cropped/{img_fname.split("/")[-1]}'
    cv2.imwrite(fout, warped)