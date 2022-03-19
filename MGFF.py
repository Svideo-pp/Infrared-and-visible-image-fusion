import numpy as np
import cv2
import argparse
import math

ITERATION = 4
RADIUS = 9
EPS = 10 ** 3 / (256 ** 2)


def guidedFilter(img_i, img_p, r, eps):
    wsize = int(2 * r) + 1
    meanI = cv2.boxFilter(img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanP = cv2.boxFilter(img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corr_I = cv2.boxFilter(img_i * img_i, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    corrIP = cv2.boxFilter(img_i * img_p, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    varI = corr_I - meanI * meanI
    covIP = corrIP - meanI * meanP
    a = covIP / (varI + eps)
    b = meanP - a * meanI
    meanA = cv2.boxFilter(a, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    meanB = cv2.boxFilter(b, ksize=(wsize, wsize), ddepth=-1, normalize=True)
    q = meanA * img_i + meanB
    return q


def MGFF_GRAY(img_r, img_v):
    img_r_base_pre = img_r * 1. / 255
    img_v_base_pre = img_v * 1. / 255
    img_r_detail = []
    img_v_detail = []
    for i in range(ITERATION):
        img_r_base_cur = guidedFilter(img_v_base_pre, img_r_base_pre, RADIUS, EPS)
        img_v_base_cur = guidedFilter(img_r_base_pre, img_v_base_pre, RADIUS, EPS)
        img_r_detail.append(img_r_base_pre - img_r_base_cur)
        img_v_detail.append(img_v_base_pre - img_v_base_cur)
        img_r_base_pre = img_r_base_cur
        img_v_base_pre = img_v_base_cur
    img_base_fused = (img_r_base_pre + img_v_base_pre) / 2
    fused_img = img_base_fused
    for i in range(ITERATION - 1, -1, -1):
        weights = np.abs(img_r_detail[i]) / (np.abs(img_r_detail[i]) + np.abs(img_v_detail[i]))
        fused_img += weights * img_r_detail[i] + (1 - weights) * img_v_detail[i]
    fused_img = cv2.normalize(fused_img, None, 0., 255., cv2.NORM_MINMAX)
    fused_img = cv2.convertScaleAbs(fused_img)
    return fused_img


def MGFF_RGB(img_r, img_v):
    r_R = img_r[:, :, 2]
    r_G = img_r[:, :, 1]
    r_B = img_r[:, :, 0]
    v_R = img_v[:, :, 2]
    v_G = img_v[:, :, 1]
    v_B = img_v[:, :, 0]
    fused_R = MGFF_GRAY(r_R, v_R)
    fused_G = MGFF_GRAY(r_G, v_G)
    fused_B = MGFF_GRAY(r_B, v_B)
    fused_img = np.stack((fused_B, fused_G, fused_R), axis=-1)
    return fused_img


def preprocess(img_ir, img_vis):
    '''

    :param img_ir: a numpy array representing the infrared image
    :param img_vis: a numpy array representing the visible image
    :param isPi: to distinguish whether running on pc or raspberry pi
    :return: two numpy array in the same size representing the IR and VIS image
    (the size will decided by infrared image)
    '''
    # img_vis: (480, 640, 3) <class 'numpy.ndarray'>
    # img_ir: ()
    # img_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    img_vis_cropped = img_vis[60 + 20:420 + 20, 80 - 20:560 - 20, :] # (360, 480, 3) <class 'numpy.ndarray'>
    img_vis_resized = cv2.resize(img_vis_cropped, (320, 250), interpolation=cv2.INTER_NEAREST)
    img_vis_resized = img_vis_resized[:240][...][...]  # (240, 320, 3)
    cv2.imshow('img_vis_preprocessed', img_vis_resized)
    cv2.imshow('img_ir_preprocessed', img_ir)
    return img_ir, img_vis_resized


def MGFF(r_path, v_path):
    img_r = cv2.imread(r_path)
    img_v = cv2.imread(v_path)
    if not isinstance(img_r, np.ndarray):
        print("img_r is not an image")
        return
    if not isinstance(img_v, np.ndarray):
        print("img_v is not an image")
        return
    img_r, img_v = preprocess(img_r, img_v)
    if img_r.shape[0] != img_v.shape[0] or img_r.shape[1] != img_v.shape[1]:
        print('size is not equal')
        return
    fused_img = None
    if len(img_r.shape) == 2 or img_r.shape[-1] == 1:
        if img_v.shape[-1] == 3:
            img_v = cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY)
        fused_img = MGFF_GRAY(img_r, img_v)
    else:
        if img_r.shape[-1] == 3:
            fused_img = MGFF_RGB(img_r, img_v)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            fused_img = MGFF_GRAY(img_r, img_v)
    cv2.imshow("fused image", fused_img)
    cv2.imwrite("fused_image_mgff.jpg", fused_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IR", default='C:\\Users\\svideo\\Desktop\\FLIR/FLIR0037.jpg', help="path to IR image", required=False)
    parser.add_argument("--VIS", default='C:\\Users\\svideo\\Desktop\\FLIR/FLIR0038.jpg', help="path to IR image", required=False)
    a = parser.parse_args()
    MGFF(a.IR, a.VIS)