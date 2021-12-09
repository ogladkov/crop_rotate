import cv2
import numpy as np
import os


def crop(img_fname, msk_fname):
    if not os.path.exists('./cropped'):
        os.mkdir('./cropped')

    img = cv2.imread(img_fname)
    msk = cv2.imread(msk_fname, 0)

    ret, thresh = cv2.threshold(msk, 127, 255, 0)
    kernel = np.ones((35,35))
    thresh = cv2.erode(thresh, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    msk = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(msk)
    box = np.int0(box)
    pts = []
    for p in box:
        pt = [p[0], p[1]]
        pts.append(pt)

    xsorted = np.array(pts)[np.argsort(np.array(pts)[:, 0])]
    lefts = xsorted[:2];
    rights = xsorted[2:]
    tl, bl = lefts[np.argsort(lefts[:, 1])]
    tr, br = rights[np.argsort(rights[:, 1])]
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    pts = np.float32([tl, tr, br, bl])

    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    fout = f'./cropped/{img_fname.split("/")[-1]}'
    cv2.imwrite(fout, warped)